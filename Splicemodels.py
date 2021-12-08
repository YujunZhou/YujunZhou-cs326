from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pickle
import copy


# the model parameters for each model
Splice_Model = {
    'Normal': './classifier/Adam_RNN.4832',
    'adversarial': './classifier/Adam_RNN.17490',
    'gnn': './classifier/gnnF.1916',
    'gnnadv': './classifier/gnnFadvpcsc.4904',
    'gnnlstm': './classifier/gnnlstm.15858',
    'gnnlstmadv': './classifier/gnnlstmadvsc.19063',
}

IPS_Model = {
    'Normal': './classifier/Mal_RNN.942',
    'adversarial': './classifier/Mal_adv.705',
    'gnn': '././classifier/mal_gnnF.846',
    'gnnadv': './classifier/mal_gnnFadvsc.577',
    'gnnlstm': './classifier/mal_gnnlstm.10078',
    'gnnlstmadv': './classifier/mal_gnnlstmadvsnc.18680',
}

Model = {
    'Splice': Splice_Model,
    'IPS': IPS_Model,
}


# return the model parameter
def model_file(Dataset, Model_Type):
    return Model[Dataset][Model_Type]


class SpliceLSTM(nn.Module):
    def __init__(self):
        super(SpliceLSTM, self).__init__()
        dropout_rate = 0.5
        input_size = 30
        self.hidden_size = 30
        n_labels = 3
        self.num_layers = 4
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(self.hidden_size, n_labels)
        self.attention = SelfAttention(self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.n_diagnosis_codes = 5
        self.embed = torch.nn.Embedding(self.n_diagnosis_codes, 30)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    # overload forward() method
    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.embed(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).relu().mean(dim=2)

        h0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        c0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        x = self.dropout(x)

        logit = self.fc(x)

        logit = self.softmax(logit)

        return logit


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.num_features = 60
        self.n_diagnosis_codes = 5
        self.emb_size = 5
        self.emb = nn.Embedding(self.n_diagnosis_codes, self.emb_size)
        self.hidden_size = 10
        self.hidden_channels = 15
        self.num_classes = 3
        self.num_new_features = 20
        self.fc = nn.Linear(self.num_features, self.num_new_features)
        self.fc2 = nn.Linear(self.num_new_features * self.emb_size, self.hidden_size)
        self.conv1 = SAGEConv(self.hidden_size, self.hidden_channels)
        self.conv2 = SAGEConv(self.hidden_channels, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))


    def forward(self, x, y):
        x = x.permute(1, 0, 2)
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.relu()
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.relu()
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), -1)
        x = self.fc2(x)
        x = x.relu()

        # find the closest 5 neighbors and make an edge between them
        label_mat = torch.mm(y.float(), y.float().t())
        label_mat = torch.where(label_mat > 0, torch.ones_like(label_mat), label_mat)
        a = torch.matmul(x, x.t())
        norm = torch.norm(x, 2, 1).reshape(1, -1)
        a = a / norm / norm.t()
        a = a * label_mat
        top5 = a.topk(6, dim=1)[1][:, 1:]
        A = torch.eye(len(x))[top5].sum(dim=1)
        A = A + A.t()
        edges = torch.cat(torch.where(A > 0), dim=0).reshape(2, -1).cuda()

        out = self.conv1(x, edges)
        out = out.relu()
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.conv2(out, edges)
        out2 = self.linear(out)
        out2 = self.softmax(out2)
        return out2, x, out, edges


class GNNtest(GNN):
    def __init__(self, model_name):
        super(GNNtest, self).__init__()
        self.x_0 = pickle.load(open('./gnn/Splice' + model_name + '.nodes.pickle', 'rb')).cuda()
        self.edge_0 = pickle.load(open('./gnn/Splice' + model_name + '.edges.pickle', 'rb')).cuda()

    def forward(self, x):
        x = x.permute(1, 0, 2)
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.relu()
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.relu()
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), -1)
        x = self.fc2(x)
        x = x.relu()

        # find the closest 5 neighbors and make an edge between them
        a = torch.matmul(x, self.x_0.t())
        norm = torch.norm(x, 2, 1).reshape(-1, 1)
        norm_0 = torch.norm(self.x_0, 2, 1).reshape(1, -1)
        a = a / norm / norm_0
        top6 = a.topk(6, dim=1)[1]
        top6_value = a.topk(6, dim=1)[0]
        temp = torch.where(top6_value[:, 0] == 1.0, torch.ones(len(top6)).cuda(), torch.zeros(len(top6)).cuda())
        top5 = torch.zeros(len(top6), 5).long().cuda()
        for i in range(len(top6)):
            if temp[i] > 0:
                top5[i] = top6[i, 1:]
            else:
                top5[i] = top6[i, :-1]
        A = torch.eye(len(self.x_0))[top5].sum(dim=1)
        edges = torch.cat(torch.where(A > 0), dim=0).reshape(2, -1).cuda()
        edges[0] += 3190
        edges = torch.cat((edges[1], edges[0])).reshape(2, -1)
        x = torch.cat((self.x_0, x), dim=0)
        edges = torch.cat((self.edge_0, edges), dim=1)

        out = self.conv1(x, self.edge_0)
        out = out.relu()
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.conv2(out, edges)
        out = self.linear(out)
        out = self.softmax(out)
        return out[len(self.x_0):]


class GNNadv(GNN):
    def __init__(self, model_name):
        super(GNNadv, self).__init__()
        self.edge_0 = pickle.load(open('./gnn/Splice' + model_name + '.edges.pickle', 'rb')).cuda()
        self.n = 3190

    def forward(self, x, y, fixedge=False):
        x = x.permute(1, 0, 2)
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.relu()
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.relu()
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), -1)
        x = self.fc2(x)
        x = x.relu()

        if fixedge:
            edges = self.edge_0
        else:
            # find the closest 5 neighbors and make an edge between them
            label_mat = torch.mm(y.float(), y.float().t())
            label_mat = torch.where(label_mat > 0, torch.ones_like(label_mat), label_mat)
            a = torch.matmul(x[:self.n], x[:self.n].t())
            norm = torch.norm(x[:self.n], 2, 1).reshape(1, -1)
            a = a / norm / norm.t()
            a = a * label_mat
            top5 = a.topk(6, dim=1)[1][:, 1:]
            A = torch.eye(len(x[:self.n]))[top5].sum(dim=1)
            A = A + A.t()
            edges = torch.cat(torch.where(A > 0), dim=0).reshape(2, -1).cuda()
            self.edge_0 = edges
        # add extra edges based on the constraints of adversarial graph structure
        edges1 = copy.deepcopy(edges)
        edges1 += self.n
        edges2 = torch.LongTensor([range(self.n), range(self.n)]).cuda()
        edges3 = copy.deepcopy(edges2)
        edges2[1] += self.n
        edges3[0] += self.n
        edges_all = torch.cat((edges, edges1, edges2, edges3), dim=1)

        out = self.conv1(x, edges_all)
        out = out.relu()
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.conv2(out, edges)
        out2 = self.linear(out)
        out2 = self.softmax(out2)
        return out2, x, out, self.edge_0


class GNNLSTM(nn.Module):
    def __init__(self):
        super(GNNLSTM, self).__init__()
        self.num_features = 60
        self.n_diagnosis_codes = 5
        self.emb_size = 20
        self.emb = nn.Embedding(self.n_diagnosis_codes, self.emb_size)
        self.hidden_size = 20
        self.hidden_channels = 15
        self.num_classes = 3
        self.conv1 = SAGEConv(self.hidden_size, self.hidden_channels)
        self.conv2 = SAGEConv(self.hidden_channels, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))
        self.input_size = 20
        self.num_layers = 4
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=False)
        self.attention = SelfAttention(self.hidden_size)

    def forward(self, x, y):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.relu()
        output, h_n = self.lstm(x)
        x, attn_weights = self.attention(output.transpose(0, 1))
        x = x.reshape(x.size(0), -1)

        # find the closest 5 neighbors and make an edge between them
        label_mat = torch.mm(y.float(), y.float().t())
        label_mat = torch.where(label_mat > 0, torch.ones_like(label_mat), label_mat)
        a = torch.matmul(x, x.t())
        norm = torch.norm(x, 2, 1).reshape(1, -1)
        a = a / norm / norm.t()
        a = a * label_mat
        top5 = a.topk(6, dim=1)[1][:, 1:]
        A = torch.eye(len(x))[top5].sum(dim=1)
        A = A + A.t()
        edges = torch.cat(torch.where(A > 0), dim=0).reshape(2, -1).cuda()

        out = self.conv1(x, edges)
        out = out.relu()
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.conv2(out, edges)
        out2 = self.linear(out)
        out2 = self.softmax(out2)
        return out2, x, out, edges


class GNNLSTMtest(GNNLSTM):
    def __init__(self, model_name, FSGS=True):
        super(GNNLSTMtest, self).__init__()
        self.x_0 = pickle.load(open('./gnn/Splice' + model_name + '.nodes.pickle', 'rb')).cuda()
        self.edge_0 = pickle.load(open('./gnn/Splice' + model_name + '.edges.pickle', 'rb')).cuda()

    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.relu()
        output, h_n = self.lstm(x)
        x, attn_weights = self.attention(output.transpose(0, 1))
        x = x.reshape(x.size(0), -1)

        # find the closest 5 neighbors and make an edge between them
        a = torch.matmul(x, self.x_0.t())
        norm = torch.norm(x, 2, 1).reshape(-1, 1)
        norm_0 = torch.norm(self.x_0, 2, 1).reshape(1, -1)
        a = a / norm / norm_0
        top6 = a.topk(6, dim=1)[1]
        top6_value = a.topk(6, dim=1)[0]
        temp = torch.where(top6_value[:, 0] == 1.0, torch.ones(len(top6)).cuda(), torch.zeros(len(top6)).cuda())
        top5 = torch.zeros(len(top6), 5).long().cuda()
        for i in range(len(top6)):
            if temp[i] > 0:
                top5[i] = top6[i, 1:]
            else:
                top5[i] = top6[i, :-1]
        A = torch.eye(len(self.x_0))[top5].sum(dim=1)
        edges = torch.cat(torch.where(A > 0), dim=0).reshape(2, -1).cuda()
        edges[0] += 3190
        edges = torch.cat((edges[1], edges[0])).reshape(2, -1)
        x = torch.cat((self.x_0, x), dim=0)
        edges = torch.cat((self.edge_0, edges), dim=1)

        out = self.conv1(x, self.edge_0)
        out = out.relu()
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.conv2(out, edges)
        out = self.linear(out)
        out = self.softmax(out)
        return out[len(self.x_0):]


class GNNLSTMadv(GNNLSTM):
    def __init__(self, model_name):
        super(GNNLSTMadv, self).__init__()
        self.edge_0 = pickle.load(open('./gnn/Splice' + model_name + '.edges.pickle', 'rb')).cuda()
        self.n = 3190
    def forward(self, x, y, fixedge=False):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.emb(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).sum(dim=2)
        x = x.relu()
        output, h_n = self.lstm(x)
        x, attn_weights = self.attention(output.transpose(0, 1))
        x = x.reshape(x.size(0), -1)
        if fixedge:
            edges = self.edge_0
        else:
            # find the closest 5 neighbors and make an edge between them
            label_mat = torch.mm(y.float(), y.float().t())
            label_mat = torch.where(label_mat > 0, torch.ones_like(label_mat), label_mat)
            a = torch.matmul(x[:self.n], x[:self.n].t())
            norm = torch.norm(x[:self.n], 2, 1).reshape(1, -1)
            a = a / norm / norm.t()
            a = a * label_mat
            top5 = a.topk(6, dim=1)[1][:, 1:]
            A = torch.eye(len(x[:self.n]))[top5].sum(dim=1)
            A = A + A.t()
            edges = torch.cat(torch.where(A > 0), dim=0).reshape(2, -1).cuda()
            self.edge_0 = edges

        # add extra edges based on the constraints of adversarial graph structure
        edges1 = copy.deepcopy(edges)
        edges1 += self.n
        edges2 = torch.LongTensor([range(self.n), range(self.n)]).cuda()
        edges3 = copy.deepcopy(edges2)
        edges2[1] += self.n
        edges3[0] += self.n
        edges_all = torch.cat((edges, edges1, edges2, edges3), dim=1)
        out = self.conv1(x, edges_all)
        out = out.relu()
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.conv2(out, edges_all)
        out2 = self.linear(out)
        out2 = self.softmax(out2)
        return out2, x, out, self.edge_0


# the model to capture the final feature for LSTM
class SpliceLSTM_temp(nn.Module):
    def __init__(self):
        super(SpliceLSTM_temp, self).__init__()
        dropout_rate = 0.5
        input_size = 30
        self.hidden_size = 30
        n_labels = 3
        self.num_layers = 4
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(self.hidden_size, n_labels)
        self.attention = SelfAttention(self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.n_diagnosis_codes = 5
        self.embed = torch.nn.Embedding(self.n_diagnosis_codes, 30)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    # overload forward() method
    def forward(self, x):
        model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.embed(model_input)
        x = torch.unsqueeze(x, dim=3)
        x = (x * weight).relu().mean(dim=2)

        h0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        c0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        return x









