import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D, bn_decay, T):
        super(STEmbedding, self).__init__()
        self.T = T
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_te = FC(
            input_dims=[T+7+2, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE, TE):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        T = self.T
        # temporal embedding (:,:,7) (:,:,288)
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(SE.device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(SE.device)
        is_holiday = torch.empty(TE.shape[0], TE.shape[1], 2).to(SE.device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T, T)
        for j in range(TE.shape[0]):
            is_holiday[j] = F.one_hot(TE[..., 2][j].to(torch.int64) % 2, 2)

        TE = torch.cat((dayofweek, timeofday, is_holiday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = self.FC_te(TE)
        del dayofweek, timeofday
        return torch.add(SE, TE)



class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # fix reproduction code issue; [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            # mask = mask.to(torch.bool)
            mask = mask.bool().int()
            mask_rev = -(mask - 1)
            mask_rev = mask_rev.to(X.device)
            mask = mask.to(X.device)
            attention = mask * attention + mask_rev * torch.full(attention.shape, -2 ** 15 + 1, device=X.device)
        # softmax
        attention = torch.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention, mask, mask_rev
        return X


class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay, global_att=True):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.global_att = global_att
        self.cst = 10e-6
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k1 = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v1 = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_q1 = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=[D,D], units=[D,D], activations=[F.relu,None],
                     bn_decay=bn_decay)
        # hop = 2 in this setting
        self.hopwise = Parameter(torch.ones(2+1, dtype=torch.float))
        self.teleport = Parameter(torch.ones(1, dtype=torch.float))
        
    def forward(self, X, STE, adj1, adj2):
        batch_size = X.shape[0]
        # [B, T, N, 2D]
        X = torch.cat((X, STE), dim=-1)

        # calcluate attention between nodes from same time step [batch_size, num_step, num_vertex, K * d=D]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # fixed issus: change self.K to d; [K * B, T, N, d];in torch split(xx,d,xx) means d elements in a sub-set
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        query = F.relu(query)
        key = F.relu(key)

        hidden = value*self.hopwise[0]
        # [K * B, T, N, d]
        B, T, N, D = query.shape

        if self.global_att:
            teleM = torch.matmul(key.transpose(2,3),value)/N
            teleM = teleM.unsqueeze(-3)
            teleK = torch.sum(key,dim=-2,keepdim=True)/N
            teleH = torch.einsum('btnd,btndz->btnz',[query,teleM])
            teleC = torch.einsum('btnd,btnd->btn', [query,teleK]).unsqueeze(-1)+self.cst

            teleH = teleH/teleC
            hidden += teleH*self.teleport
        
        del teleM, teleH, teleC

        A = torch.cat([torch.cat([adj1,torch.eye(N).to(adj1.device)]),torch.cat([torch.eye(N).to(adj1.device),adj1])],dim=-1)
        edge_index = torch.where(A != 0)
        row, col = edge_index
        deg = degree(col,2*N, dtype=query.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row]

        # To compute two-hop attention
        X2 = torch.cat((X[:,0,:,:].unsqueeze(1), X),dim=1) # padding 0 at frist
        # print(X2.shape)
        key2 = []
        query2 = []
        value2 = []
        # print(num_step)
        for i in range(T):
            # [B, T, N, K*d]
            # print(X2[:,i:i+2,:,:].reshape(batch_size,2*num_node,-1).unsqueeze(1).shape)
            key2.append(X2[:,i:i+2,:,:].reshape(batch_size,2*N,-1).unsqueeze(1))
            query2.append(X2[:,i:i+2,:,:].reshape(batch_size,2*N,-1).unsqueeze(1))
            value2.append(X2[:,i:i+2,:,:].reshape(batch_size,2*N,-1).unsqueeze(1))
        key2 = torch.cat(key2, dim=1)
        query2 = torch.cat(query2, dim=1)
        value2 = torch.cat(value2, dim=1)
        key2 = self.FC_k1(key2)
        query2 = self.FC_q1(query2)
        value2 = self.FC_v1(value2)
        K = torch.cat(torch.split(key2, self.d, dim=-1), dim=0)
        query2 = torch.cat(torch.split(query2, self.d, dim=-1), dim=0)
        value2 = torch.cat(torch.split(value2, self.d, dim=-1), dim=0)

        query2 = F.relu(query2)
        K = F.relu(K)

        M = torch.einsum('btnd,btnz->btndz',[K,value2])

        for hop in range(2):
            M_j = M[:,:,row,:,:]
            M_j = norm.view(-1,1,1)*M_j
            M = scatter(M_j, col, dim=-3, reduce='sum')
            K_j = K[:,:,row,:]
            K_j = norm.view(-1,1)*K_j
            K = scatter(K_j, col, dim=-2, reduce='sum')
            H = torch.einsum('btnd,btndz->btnz',[query2,M])
            C = torch.einsum('btnd,btnd->btn',[query2,K]).unsqueeze(-1) + self.cst
            H = H/C

            hidden += H[:,:,N:,:]*self.hopwise[hop+1]

        hidden = torch.cat(torch.split(hidden, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        Z = self.FC(hidden)
        del K, H, C, M, key2, value2, key, value, query
        return Z


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STAttBlock(nn.Module):
    def __init__(self, K, d, bn_decay, mask=True):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay, global_att=True)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay)

    def forward(self, X, STE, adj1, adj2, is_train):
        HS = self.spatialAttention(X, STE, adj1, adj2)
        HT = self.temporalAttention(X, STE)

        H = self.gatedFusion(HS, HT)
        # revised comment del
        del HS, HT
        # del HS, HT, HSS
        return torch.add(X, H)


class transformAttention(nn.Module):
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        # chage D to [D,D]
        self.FC = FC(input_dims=[D,D], units=[D,D], activations=[F.relu, None],
                     bn_decay=bn_decay)
        self.cst = 10e-6

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = F.relu(query)
        key = F.relu(key)
        # query: [K * batch_size, N, T, d]
        # key:   [K * batch_size, N, d, T]
        # value: [K * batch_size, N, T, d]
        query = query.permute(0, 2, 1, 3) 
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        T = query.shape[-2]
        M = torch.matmul(key, value)/T
        H = torch.matmul(query, M)/T
        C = torch.matmul(query, torch.sum(key,dim=-1,keepdim=True)/T)+self.cst
        H = H/C
        H = H.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(H, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, M, H, C
        return X


class GMAN(nn.Module):
    '''
    GMAN
        X：       [batch_size, num_his, num_vertx]
        TE：      [batch_size, num_his + num_pred, 2] (time-of-day, day-of-week)
        SE：      [num_vertex, K * d]
        num_his： number of history steps
        num_pred：number of prediction steps
        T：       one day is divided into T steps (288 for PEMS)
        L：       number of STAtt blocks in the encoder/decoder
        K：       number of attention heads
        d：       dimension of each attention head outputs
        return：  [batch_size, num_pred, num_vertex]
    '''

    def __init__(self, SE, adj1, adj2, num_prototypes, args, bn_decay):
        super(GMAN, self).__init__()
        L = args.L
        K = args.K
        d = args.d
        D = K * d
        self.num_his = args.num_his
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.SE = SE
        self.adj1 = adj1
        self.adj2 = adj2
        N = adj1.shape[0]
        self.num_pred = args.num_pred
        self.num_prototypes = num_prototypes
        self.slot_per_h = args.T/24
        # self.update_rate = args.update_rate
        self.teleport = Parameter(torch.ones(1, dtype=torch.float),requires_grad=True)
        self.anchors = Parameter(torch.rand(num_prototypes, args.num_pred, N, args.K * args.d),requires_grad=True)
        # initial edges mask equal to 0.5
        # self.edge_m1 = nn.Parameter(torch.ones(self.adj1.shape)*0.5,requires_grad=True)
        # self.edge_m2 = nn.Parameter(torch.ones(self.adj2.shape)*0.5,requires_grad=True)
       
        self.STEmbedding = STEmbedding(D, bn_decay,args.T)

        # embeding the input and generate the attention score (filter)
        self.FC_1 = FC(input_dims=[self.input_dim, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.node_filter = FC(input_dims=D, units=2, activations=[None],
                       bn_decay=bn_decay)

        # two branches: for invariant branch
        self.STAttBlock_IF = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STAttBlock_IB = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttentionI = transformAttention(K, d, bn_decay)

        # for environment branch
        self.STAttBlock_EF = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STAttBlock_EB = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttentionE = transformAttention(K, d, bn_decay)

        # merge two branches
        # self.weights = nn.Parameter(torch.ones(1, dtype=torch.float,requires_grad=True))
        
        self.FC_I = FC(input_dims=self.num_pred, units=1, activations=None,
                       bn_decay=bn_decay)
        self.FC_E = FC(input_dims=self.num_pred, units=1, activations=None,
                       bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[2*D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_Z = FC(input_dims=[D, D], units=[D, self.output_dim], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, X, TE, is_train=True):
        # input
        if len(X.shape) == 3:
            X = torch.unsqueeze(X, -1)
        B = X.shape[0]
        X = self.FC_1(X)
        # [batch_size, num_his, num_vertex, D]; compute attention weights for speratete node features
        atten = self.node_filter(X)
        atten = torch.softmax(atten, dim=-1)
        atten_i = atten[:, :, :, 0]
        atten_e = atten[:, :, :, 1]
        X_i = X * torch.unsqueeze(atten_i, -1)
        X_e = X * torch.unsqueeze(atten_e, -1)

        # STE [B, T, N, D]
        STE = self.STEmbedding(self.SE, TE)
        
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]

        # two branches
        # invariant branch
        for net in self.STAttBlock_IF:
            X_i = net(X_i, STE_his, self.adj1, self.adj2, is_train)
        X_i = self.transformAttentionI(X_i, STE_his, STE_pred)
        for net in self.STAttBlock_IB:
            X_i = net(X_i, STE_pred, self.adj1, self.adj2, is_train)

        # environment branch
        for net in self.STAttBlock_EF:
            X_e = net(X_e, STE_his, self.adj1, self.adj2, is_train)
        X_e = self.transformAttentionE(X_e, STE_his, STE_pred)
        for net in self.STAttBlock_EB:
            X_e = net(X_e, STE_pred, self.adj1, self.adj2, is_train)

        # 1. for invariant branch; weekend day 1,2; working day 3-6; TE [B, T, D]-dow; tod
        # anchors = self.anchors
        TE_his = TE[:, :self.num_his, :]
        dayofweek = TE_his[..., 0] # B, T
        timeofday = TE_his[..., 1]
        is_holiday = TE_his[..., 2]
        
        p_Z = [[] for i in range(self.num_prototypes)]
        z = torch.zeros(X_i.shape).to(X_i.device)
        # 0-14; wokring day morning rush hour; working day evening rush hour; working day non-rush hour; Sat; Sun
        for i in range(B):
            if is_holiday[i, 0] == 1:
                if dayofweek[i, 1] == 1:
                    p_Z[-2].append(X_i[i, ...]) # Sat
                    z[i] = self.anchors[-2]
                else:
                    p_Z[-1].append(X_i[i, ...]) # Sun or public holiday
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
                    p_Z[d*3].append(X_i[i, ...])
                    z[i] = self.anchors[d*3]
                # evening rush hour: p2
                elif timeofday[i, 0] >= 16*self.slot_per_h and timeofday[i, 0] < 22*self.slot_per_h:
                    p_Z[d*3+1].append(X_i[i, ...])
                    z[i] = self.anchors[d*3+1]
                else:
                    p_Z[d*3+2].append(X_i[i, ...])
                    z[i] = self.anchors[d*3+2]

        # merge two branches by cat [B, T, N, D]
        X = torch.cat((X_i, X_e), dim=-1)
        # output
        X = self.FC_2(X)
        X = X + z * self.teleport
        X = self.FC_Z(X)


        # graphs representation; [B, T, N, D] - [B, D, N, T] - [B, 1, N, D]
        g_feati = X_i.clone().permute(0, 3, 2, 1)
        g_feate = X_e.clone().permute(0, 3, 2, 1)
        g_feati = self.FC_I(g_feati)
        g_feate = self.FC_E(g_feate)
        # [B, 1, N, D]
        g_feati = g_feati.permute(0, 3, 2, 1)
        g_feate = g_feate.permute(0, 3, 2, 1)

        # For KL divergence
        g_feati = torch.mean(g_feati, dim=-2)
        g_feate = torch.mean(g_feate, dim=-2)

        g_feati = g_feati.squeeze()
        g_feate = g_feate.squeeze()
        
        del STE, STE_his, STE_pred
        return g_feati, g_feate, p_Z, self.anchors, torch.squeeze(X, 3)

        