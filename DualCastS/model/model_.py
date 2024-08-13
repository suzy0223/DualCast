from logging import getLogger
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter

class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        self.bn_decay = bn_decay
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        decay = self.bn_decay if self.bn_decay is not None else 0.1
        self.batch_norm = nn.BatchNorm2d(output_dims, eps=1e-3, momentum=decay)
        if self.activation is not None:
            self.relu = nn.ReLU()
        
        self.initialize_weights(use_bias)
    
    def initialize_weights(self, use_bias=True):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(self.conv.weight)
                # nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if use_bias:
                    nn.init.constant_(self.conv.bias, 0)
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                # nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')


    def forward(self, x):
        # pytorch conv2: B,C,H,W
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.relu(x)

        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
    
    
class SSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.values2 = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys2 = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries2 = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)
        
        self.hopwise = Parameter(torch.ones(2+1, dtype=torch.float))
        self.teleport = Parameter(torch.ones(1, dtype=torch.float))
        self.cst = 10e-6
        self.global_att=True

    def forward(self, values, keys, query, adj=None):
        # B N T D
        batch_size, N, input_window, embed_dim = query.shape
        
        # B N T D -> B T N D
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        
        v2 = torch.cat((values[:,0,:,:].unsqueeze(1), values),dim=1) # padding 0 at frist
        k2 = torch.cat((keys[:,0,:,:].unsqueeze(1), keys),dim=1) # padding 0 at frist
        q2 = torch.cat((query[:,0,:,:].unsqueeze(1), query),dim=1) # padding 0 at frist

        # B T N k d -> B T k N d
        values = values.reshape(batch_size, input_window, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        keys = keys.reshape(batch_size, input_window, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        query = query.reshape(batch_size, input_window, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        # B N T k d
        queries = F.relu(queries)
        keys = F.relu(keys)
        
        hidden = values*self.hopwise[0]
        if self.global_att:
            teleM = torch.matmul(keys.transpose(3,4),values)/N
            teleM = teleM.unsqueeze(-3)
            teleK = torch.sum(keys,dim=-2,keepdim=True)/N
            teleH = torch.einsum('bthnd,bthndz->bthnz',[queries,teleM])
            teleC = torch.einsum('bthnd,bthnd->bthn', [queries,teleK]).unsqueeze(-1)+self.cst
            teleH = teleH/teleC
            hidden = hidden + teleH*self.teleport[0]
        
        if adj is not None:
            A = torch.cat([torch.cat([adj,torch.eye(N).to(adj.device)]),torch.cat([torch.eye(N).to(adj.device),adj])],dim=-1)
        else:
            A = torch.cat([torch.cat([torch.eye(N),torch.eye(N)]),torch.cat([torch.eye(N),torch.eye(N)])],dim=-1).to(adj.device)
        # print(A)
        edge_index = torch.where(A != 0)
        row, col = edge_index
        deg = degree(col,2*N, dtype=queries.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row]

        # To compute two-hop attention
        keys2 = []
        values2 = []
        query2 = []
        # print(num_step)
        for i in range(input_window):
            # [B, 1, 2N, D] 
            keys2.append(k2[:,i:i+2,:,:].reshape(batch_size, 2*N, -1).unsqueeze(1))
            values2.append(v2[:,i:i+2,:,:].reshape(batch_size, 2*N, -1).unsqueeze(1))
            query2.append(q2[:,i:i+2,:,:].reshape(batch_size, 2*N, -1).unsqueeze(1))

        # [B, T, 2N, D] -> (B,T,2N,K,d) -> (B,T,K,2N,d)
        keys2 = torch.cat(keys2, dim=1).reshape(batch_size, input_window, 2*N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        values2 = torch.cat(values2, dim=1).reshape(batch_size, input_window, 2*N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        query2 = torch.cat(query2, dim=1).reshape(batch_size, input_window, 2*N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        
        keys2 = self.keys2(keys2)
        values2 = self.values2(values2)
        query2 = self.queries2(query2)
        
        query2 = F.relu(query2)
        keys2 = F.relu(keys2)
        M = torch.einsum('btknd,btknz->btkndz',[keys2,values2])

        # print(len(row))
        # print(M.shape)

        for hop in range(2):
            M_j = M[:,:,:,row,:,:]
            M_j = norm.view(-1,1,1)*M_j
            M = scatter(M_j, col, dim=-3, reduce='sum')

            K_j = keys2[:,:,:,row,:]
            K_j = norm.view(-1,1)*K_j
            keys2 = scatter(K_j, col, dim=-2, reduce='sum')
            H = torch.einsum('btknd,btkndz->btknz',[query2,M])
            C = torch.einsum('btknd,btknd->btkn',[query2,keys2]).unsqueeze(-1) + self.cst
            H = H/C

            hidden += H[:,:,:,N:,:]*self.hopwise[hop+1]
        
        # B T K N d -> B T N K d -> B T N D
        hidden = hidden.transpose(2, 3).reshape(batch_size, input_window, N, self.num_heads * self.head_dim)
        # B N T D
        hidden = hidden.permute(0, 2, 1, 3)

        out = self.fc_out(hidden)

        return out


class TSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dim needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)

    def forward(self, values, keys, query):
        batch_size, num_nodes, input_window, embed_dim = query.shape

        # B N T k d
        values = values.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, num_nodes, input_window, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # B N T k d * B N T k d = B N T T k
        energy = torch.einsum("bnqhd,bnkhd->bnqkh", [queries, keys])
        
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        # B N T T k * B N T k d = B N T k d -> B N T D
        out = torch.einsum("bnqkh,bnkhd->bnqhd", [attention, values]).reshape(
            batch_size, num_nodes, input_window, self.num_heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=torch.device('cpu')):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features).to(device))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_mx):
        support = torch.einsum("bnd, dh->bnh", [x, self.weight])
        output = torch.einsum("mn,bnh->bmh", [adj_mx, support])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, device=device)
        self.gc2 = GraphConvolution(nhid, nclass, device=device)
        self.dropout_rate = dropout_rate

    def forward(self, x, adj_mx):
        x = F.relu(self.gc1(x, adj_mx))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc2(x, adj_mx)
        return F.log_softmax(x, dim=2)


class STransformer(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_heads=1,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.adj_mx = adj_mx
        self.adj_mx_org = self.adj_mx.clone()
        # self.adj_mx = torch.FloatTensor(adj_mx).to(device)
        self.D_S = nn.Parameter(adj_mx)
        self.embed_linear = nn.Linear(adj_mx.shape[0], embed_dim)

        self.attention = SSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

        self.gcn = GCN(embed_dim, embed_dim * 2, embed_dim, dropout_rate, device=device)
        self.norm_adj = nn.InstanceNorm2d(1)

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.fs = nn.Linear(embed_dim, embed_dim)
        self.fg = nn.Linear(embed_dim, embed_dim)

    def forward(self, value, key, query):
        batch_size, num_nodes, input_windows, embed_dim = query.shape
        D_S = self.embed_linear(self.D_S)
        D_S = D_S.expand(batch_size, input_windows, num_nodes, embed_dim)
        D_S = D_S.permute(0, 2, 1, 3)

        X_G = torch.Tensor(query.shape[0], query.shape[1], 0, query.shape[3]).to(self.device)
        adj_mx = self.adj_mx.unsqueeze(0).unsqueeze(0)
        adj_mx = self.norm_adj(adj_mx)
        adj_mx = adj_mx.squeeze(0).squeeze(0)

        for t in range(query.shape[2]):
            o = self.gcn(query[:, :, t, :], adj_mx)
            o = o.unsqueeze(2)
            X_G = torch.cat((X_G, o), dim=2)

        query = query + D_S
        attention = self.attention(value, key, query,self.adj_mx_org)

        x = self.dropout_layer(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout_layer(self.norm2(forward + x))

        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))
        out = g * U_S + (1 - g) * X_G

        return out


class TTransformer(nn.Module):
    def __init__(self, TG_per_day=288, embed_dim=64, num_heads=1,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.temporal_embedding = nn.Embedding(TG_per_day, embed_dim)

        self.attention = TSelfAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, value, key, query):
        batch_size, num_nodes, input_windows, embed_dim = query.shape

        D_T = self.temporal_embedding(torch.arange(0, input_windows).to(self.device))
        D_T = D_T.expand(batch_size, num_nodes, input_windows, embed_dim)

        query = query + D_T

        attention = self.attention(value, key, query)

        x = self.dropout_layer(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout_layer(self.norm2(forward + x))
        return out


class STTransformerBlock(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_heads=1, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.STransformer = STransformer(
            adj_mx, embed_dim=embed_dim, num_heads=num_heads,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device)
        self.TTransformer = TTransformer(
            TG_per_day=TG_per_day, embed_dim=embed_dim, num_heads=num_heads,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, value, key, query):
        x1 = self.norm1(self.STransformer(value, key, query) + query)
        x2 = self.dropout_layer(self.norm2(self.TTransformer(x1, x1, x1) + x1))
        return x2


class Encoder(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_layers=1, num_heads=1, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerBlock(
                adj_mx, embed_dim=embed_dim, num_heads=num_heads, TG_per_day=TG_per_day,
                forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device
            )
            for _ in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.dropout_layer(x)
        for layer in self.layers:
            out = layer(out, out, out)
        return out


class Transformer(nn.Module):
    def __init__(self, adj_mx, embed_dim=64, num_layers=3, num_heads=1, TG_per_day=288,
                 forward_expansion=4, dropout_rate=0, device=torch.device('cpu')):
        super().__init__()
        self.encoder = Encoder(
            adj_mx, embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads, TG_per_day=TG_per_day,
            forward_expansion=forward_expansion, dropout_rate=dropout_rate, device=device,
        )

    def forward(self, src):
        enc_src = self.encoder(src)
        return enc_src


class STTN(nn.Module):
    def __init__(self, args, adj_mx, device):
        super(STTN, self).__init__()
        
        self.adj_mx = adj_mx
        self.feature_dim = args.input_dim
        self.output_dim = args.output_dim
        
        self.embed_dim = args.embed_dim
        self.num_layers = args.L
        self.num_heads = args.K
        self.TG_per_day = args.T
        self.forward_expansion = args.forward_expansion
        self.dropout_rate = args.dropout_rate
        self.device = device
        self.input_window = args.num_his
        self.output_window = args.num_pred
        
        self.num_prototypes = args.num_prototypes
        self.slot_per_h = args.T/24

        N = self.adj_mx.shape[0]
        bn_decay=0.1
        
        self.teleport = Parameter(torch.ones(1, dtype=torch.float),requires_grad=True)
        self.anchors = Parameter(torch.rand(self.num_prototypes, args.num_pred, N, self.embed_dim),requires_grad=True)
        self.conv1 = nn.Conv2d(self.feature_dim, self.embed_dim, 1)
        
        self.node_filter = FC(input_dims=self.embed_dim, units=2, activations=[None],
                       bn_decay=bn_decay)
        
        self.transformer_I = Transformer(
            self.adj_mx, embed_dim=self.embed_dim, num_layers=self.num_layers, num_heads=self.num_heads,
            TG_per_day=self.TG_per_day, forward_expansion=self.forward_expansion, dropout_rate=self.dropout_rate,
            device=self.device,
        )
        self.transformer_E = Transformer(
            self.adj_mx, embed_dim=self.embed_dim, num_layers=self.num_layers, num_heads=self.num_heads,
            TG_per_day=self.TG_per_day, forward_expansion=self.forward_expansion, dropout_rate=self.dropout_rate,
            device=self.device,
        )
        
        self.FC_2 = FC(input_dims=[2*self.embed_dim, self.embed_dim], units=[self.embed_dim, self.embed_dim], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_Z = FC(input_dims=[self.embed_dim, self.embed_dim], units=[self.embed_dim, self.output_dim], activations=[F.relu, None],
                       bn_decay=bn_decay)

        self.conv2_i = nn.Conv2d(self.input_window, self.output_window, 1)
        self.conv2_e = nn.Conv2d(self.input_window, self.output_window, 1)

        self.conv4_i = nn.Conv2d(self.output_window, 1, 1)
        self.conv4_e = nn.Conv2d(self.output_window, 1, 1)
        
        self.act_layer = nn.ReLU()

    def forward(self, X, TE):
        # B T N D
        batch_size = X.shape[0]
        if len(X.shape) == 3:
            inputs = torch.unsqueeze(X, -1)
        else:
            inputs = X
        # B T N D -> B D N T
        inputs = inputs.permute(0, 3, 2, 1)
        # D = 1 -> D = embed_dim
        inputs = self.conv1(inputs)
        
        # B D N T -> B T N D
        inputs = inputs.permute(0, 3, 2, 1)
        atten = self.node_filter(inputs)
        atten = torch.softmax(atten, dim=-1)
        atten_i = atten[:,:,:,0]
        atten_e = atten[:,:,:,1]
        
        inputs_transformer_i = inputs * torch.unsqueeze(atten_i, -1)
        inputs_transformer_e = inputs * torch.unsqueeze(atten_e, -1)
        
        # B T N D -> B N T D
        inputs_transformer_i = inputs_transformer_i.permute(0, 2, 1, 3)
        inputs_transformer_e = inputs_transformer_e.permute(0, 2, 1, 3)
        
        output_transformer_i = self.transformer_I(inputs_transformer_i)
        output_transformer_e = self.transformer_E(inputs_transformer_e)
        
        # B N T D -> B T N D
        output_transformer_i = output_transformer_i.permute(0, 2, 1, 3)
        output_transformer_e = output_transformer_e.permute(0, 2, 1, 3)
        
        output_transformer_i = self.act_layer(self.conv2_i(output_transformer_i))
        output_transformer_e = self.act_layer(self.conv2_e(output_transformer_e))
        # print("output_transformer_i.shape = ",output_transformer_i.shape)
        
        # anchors = self.anchors
        TE_his = TE[:, :self.input_window, :]
        dayofweek = TE_his[..., 0] # B, T
        timeofday = TE_his[..., 1]
        is_holiday = TE_his[..., 2]
        
        p_Z = [[] for i in range(self.num_prototypes)]
        # B T N D
        z = torch.zeros(output_transformer_i.shape).to(output_transformer_i.device)
        # 0-14; wokring day morning rush hour; working day evening rush hour; working day non-rush hour; Sat; Sun
        for i in range(batch_size):
            if is_holiday[i, 0] == 1:
                if dayofweek[i, 1] == 1:
                    p_Z[-2].append(output_transformer_i[i, ...]) # Sat
                    z[i] = self.anchors[-2]
                else:
                    p_Z[-1].append(output_transformer_i[i, ...]) # Sun or public holiday
                    z[i] = self.anchors[-1]
            else:
                # working day non-rush hour(9-16 & 22-6)
                # moring rush hour: p1
                # if Friday
                if dayofweek[i, 0]==0:
                    d=4
                else:
                    d = dayofweek[i, 0]-3
                if timeofday[i, 0] >= 6*self.slot_per_h and timeofday[i, 0] < 9*self.slot_per_h:
                    p_Z[d*3].append(output_transformer_i[i, ...])
                    z[i] = self.anchors[d*3]
                # evening rush hour: p2
                elif timeofday[i, 0] >= 16*self.slot_per_h and timeofday[i, 0] < 22*self.slot_per_h:
                    p_Z[d*3+1].append(output_transformer_i[i, ...])
                    z[i] = self.anchors[d*3+1]
                else:
                    p_Z[d*3+2].append(output_transformer_i[i, ...])
                    z[i] = self.anchors[d*3+2]
        
        
        # # B T N D -> B D N T
        # output_transformer_i = output_transformer_i.permute(0, 3, 2, 1)
        # output_transformer_e = output_transformer_e.permute(0, 3, 2, 1)
        # output_transformer_i = self.conv3_i(output_transformer_i)
        # output_transformer_e = self.conv3_e(output_transformer_e)
        # # B D N T -> B T N D
        # output_transformer_i = output_transformer_i.permute(0, 3, 2, 1)
        # output_transformer_e = output_transformer_e.permute(0, 3, 2, 1)
        
        # B T N 2D
        out = torch.cat((output_transformer_i, output_transformer_e), dim=-1)
        # print("out.shape",out.shape)
        # B T N D
        out = self.FC_2(out)
        out = out + z * self.teleport
        # B T N D
        out = self.FC_Z(out)
        
        # B T N d -> B 1 N D
        g_feati = output_transformer_i.clone()
        g_feate = output_transformer_e.clone()
        g_feati = self.conv4_i(g_feati)
        g_feate = self.conv4_e(g_feate)
        
        # B 1 N D -> B 1 1 D -> B D
        g_feati = torch.mean(g_feati, dim=-2)
        g_feate = torch.mean(g_feate, dim=-2)
        
        g_feati = g_feati.squeeze()
        g_feate = g_feate.squeeze()
        
        del output_transformer_i, output_transformer_e
        
        out = torch.squeeze(out, 3)

        return g_feati, g_feate, p_Z, self.anchors, out
