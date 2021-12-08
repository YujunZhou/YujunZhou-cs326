import copy
import csv
import random
import time
import torch_geometric
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse
from utils import *
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='gene')  # 创建parser对象
parser.add_argument('--lr', default=0.0001, type=float, help='lr')
parser.add_argument('--algo', default='Normal', type=str, help='adversarial training or not')
parser.add_argument('--dataset', default='Splice', type=str, help='Dataset')
parser.add_argument('--model', default='linear', type=str, help='linear layer + GNN or LSTM+GNN')
args = parser.parse_args()  # 解析参数，此处args是一个命名空间列表
if args.dataset == 'Splice':
    from Splicemodels import *
else:
    from IPSmodels import *

torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)
torch_geometric.seed_everything(666)


def calculate_cost(logit, t_labels, val_mask):
    cost_sum = 0.0
    loss = F.cross_entropy(logit[val_mask], t_labels[val_mask])
    cost_sum += loss.cpu().data.numpy()
    return cost_sum


def Training(Dataset, n_epoch, lr):
    X, y = preparation(Dataset)
    N = num_samples[Dataset]

    train_idx, test_idx = train_test_split(np.array(range(N)), test_size=0.1, random_state=666)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=666)

    if adversarial:
        train_mask = np.zeros((N,), dtype=bool)
        train_mask[train_idx] = True
        train_mask = np.concatenate((train_mask, np.ones((N,), dtype=bool)))

        val_mask = np.zeros((N,), dtype=bool)
        val_mask[val_idx] = True
        val_mask = np.concatenate((val_mask, val_mask))

        test_mask = np.zeros((N,), dtype=bool)
        test_mask[test_idx] = True
    else:
        train_mask = np.zeros((N,), dtype=bool)
        train_mask[train_idx] = True

        val_mask = np.zeros((N,), dtype=bool)
        val_mask[val_idx] = True

        test_mask = np.zeros((N,), dtype=bool)
        test_mask[test_idx] = True

    output_file = './outputs/' + Dataset + '/' + Model_Name + '/' + str(lr) + '/'
    make_dir('./outputs/')
    make_dir('./outputs/' + Dataset + '/')
    make_dir('./outputs/' + Dataset + '/' + Model_Name + '/')
    make_dir('./outputs/' + Dataset + '/' + Model_Name + '/' + str(lr) + '/')
    make_dir('./gnn/')

    log_f = open(
        './Logs/' + Dataset + '/training/TEST_%s_%s.bak' % (
            Model_Name, lr), 'w+')
    print('constructing the optimizer ...', file=log_f, flush=True)

    if linear:
        if adversarial:
            model = GNNadv('gnn')
        else:
            model = GNN()
    else:
        if adversarial:
            model = GNNLSTMadv('gnnlstm')
        else:
            model = GNNLSTM()
    print(linear)
    print(adversarial)

    if adversarial:
        diagnosis_codes_adv = pickle.load(open('./dataset/'+Dataset+'_perturbed.pickle', 'rb'))
        diagnosis_codes_adv = torch.LongTensor(diagnosis_codes_adv)
        diagnosis_codes = torch.LongTensor(X)
        diagnosis_codes_all = torch.cat((diagnosis_codes, diagnosis_codes_adv), dim=0)
        y_all = torch.LongTensor(y)
        y_all = torch.cat((y_all, y_all)).cuda()
        tsne = TSNE(n_components=2)
        fig = plt.figure(figsize=(27, 250))
        fignum = 1
        if linear:
            nodes_features_cpu = pickle.load(open('./gnn/' + Dataset + 'gnn.nodes.pickle', 'rb')).cpu().data.numpy()
            gnn_features_cpu = pickle.load(open('./gnn/' + Dataset + 'gnn.gnnout.pickle', 'rb')).cpu().data.numpy()
        else:
            nodes_features_cpu = pickle.load(open('./gnn/'+Dataset+'gnnlstm.nodes.pickle', 'rb')).cpu().data.numpy()
            gnn_features_cpu = pickle.load(open('./gnn/'+Dataset+'gnnlstm.gnnout.pickle', 'rb')).cpu().data.numpy()
        nodes_tsne = tsne.fit_transform(nodes_features_cpu, y)
        gnn_tsne = tsne.fit_transform(gnn_features_cpu, y)
        nodes_min, nodes_max = nodes_tsne.min(0), nodes_tsne.max(0)
        gnn_min, gnn_max = gnn_tsne.min(0), gnn_tsne.max(0)
        nodes_norm = (nodes_tsne - nodes_min) / (nodes_max - nodes_min)
        gnn_norm = (gnn_tsne - gnn_min) / (gnn_max - gnn_min)
        ax = fig.add_subplot(22, 2, fignum)
        bx = fig.add_subplot(22, 2, fignum + 1)
        plt.setp(ax, xticks=[], yticks=[])
        plt.setp(bx, xticks=[], yticks=[])
        fignum += 2
        if linear:
            ax.title.set_text('linear, epoch 0')
        else:
            ax.title.set_text('lstm, epoch 0')
        bx.title.set_text('gnn, epoch 0')
        for i in range(nodes_norm.shape[0]):
            ax.text(nodes_norm[i, 0], nodes_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]))
        for i in range(gnn_norm.shape[0]):
            bx.text(gnn_norm[i, 0], gnn_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]))

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('done!', file=log_f, flush=True)

    print('training start', file=log_f, flush=True)

    best_train_cost = 0.0
    best_validate_cost = 10000
    epoch_duaration = 0.0
    best_epoch = 0

    for epoch in range(n_epoch):
        model.train()
        cost_vector = []
        start_time = time.time()

        diagnosis_codes = copy.deepcopy(X)
        labels = y
        if adversarial:
            diagnosis_codes = diagnosis_codes_all
            labels = y
        t_diagnosis_codes, t_labels = pad_matrix(diagnosis_codes, labels, num_category[Dataset])

        t_labels = torch.LongTensor(t_labels).cuda()
        t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()
        t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes, requires_grad=True)
        optimizer.zero_grad()

        masked_labels = one_hot_labels(t_labels, 3)
        masked_labels[test_idx] = torch.LongTensor([1, 1, 1]).cuda()
        masked_labels[val_idx] = torch.LongTensor([1, 1, 1]).cuda()
        logit, new_features, edges, gnn_features = model(t_diagnosis_codes, masked_labels)

        if adversarial:
            t_labels = y_all

        loss = F.cross_entropy(logit[train_mask], t_labels[train_mask])
        loss.backward()

        optimizer.step()

        cost_vector.append(loss.cpu().data.numpy())
        print('epoch:%d, cost:%f' % (epoch, loss.cpu().data.numpy()),
              file=log_f, flush=True)

        duration = time.time() - start_time
        train_cost = np.mean(cost_vector)
        model.eval()
        logit, new_features, gnn_features, edges = model(t_diagnosis_codes, masked_labels)
        validate_cost = calculate_cost(logit, t_labels, val_mask)
        epoch_duaration += duration

        if validate_cost < best_validate_cost:
            # torch.save(rnn.state_dict(), output_file + 'Adam_' + Model_Name + '.' + str(epoch))
            torch.save(model.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch),
                       _use_new_zipfile_serialization=False)
            pickle.dump(new_features[:N], open('./gnn/' + Dataset + Model_Name + '.nodes.pickle', 'wb'))
            pickle.dump(edges, open('./gnn/' + Dataset + Model_Name + '.edges.pickle', 'wb'))
            pickle.dump(gnn_features[:N], open('./gnn/' + Dataset + Model_Name + '.gnnout.pickle', 'wb'))
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration), file=log_f, flush=True)

        if adversarial:
            if epoch % 1 == 0:
                nodes_features_cpu = new_features[:N].cpu().data.numpy()
                gnn_features_cpu = gnn_features[:N].cpu().data.numpy()
                nodes_tsne = tsne.fit_transform(nodes_features_cpu, y)
                gnn_tsne = tsne.fit_transform(gnn_features_cpu, y)
                nodes_min, nodes_max = nodes_tsne.min(0), nodes_tsne.max(0)
                gnn_min, gnn_max = gnn_tsne.min(0), gnn_tsne.max(0)
                nodes_norm = (nodes_tsne - nodes_min) / (nodes_max - nodes_min)
                gnn_norm = (gnn_tsne - gnn_min) / (gnn_max - gnn_min)
                ax = fig.add_subplot(22, 2, fignum)
                bx = fig.add_subplot(22, 2, fignum + 1)
                plt.setp(ax, xticks=[], yticks=[])
                plt.setp(bx, xticks=[], yticks=[])
                if linear:
                    ax.title.set_text('linear, epoch %d' % (fignum // 2 * 250))
                else:
                    ax.title.set_text('lstm, epoch %d' % (fignum // 2 * 250))
                bx.title.set_text('gnn, epoch %d' % (fignum // 2 * 250))
                fignum += 2
                for i in range(nodes_norm.shape[0]):
                    ax.text(nodes_norm[i, 0], nodes_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]))
                for i in range(gnn_norm.shape[0]):
                    bx.text(gnn_norm[i, 0], gnn_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]))

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_epoch = epoch

        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f' % (best_epoch, best_train_cost, best_validate_cost)
        print(buf, file=log_f, flush=True)

    if adversarial:
        nodes_features_cpu = pickle.load(open('./gnn/' + Dataset + Model_Name + '.nodes.pickle', 'rb')).cpu().data.numpy()
        gnn_features_cpu = pickle.load(open('./gnn/' + Dataset + Model_Name + '.gnnout.pickle', 'rb')).cpu().data.numpy()
        nodes_tsne = tsne.fit_transform(nodes_features_cpu, y)
        gnn_tsne = tsne.fit_transform(gnn_features_cpu, y)
        nodes_min, nodes_max = nodes_tsne.min(0), nodes_tsne.max(0)
        gnn_min, gnn_max = gnn_tsne.min(0), gnn_tsne.max(0)
        nodes_norm = (nodes_tsne - nodes_min) / (nodes_max - nodes_min)
        gnn_norm = (gnn_tsne - gnn_min) / (gnn_max - gnn_min)
        ax = fig.add_subplot(22, 2, fignum)
        bx = fig.add_subplot(22, 2, fignum + 1)
        plt.setp(ax, xticks=[], yticks=[])
        plt.setp(bx, xticks=[], yticks=[])
        if linear:
            ax.title.set_text('linear, best epoch')
        else:
            ax.title.set_text('lstm, best epoch')
        bx.title.set_text('gnn, best epoch')
        for i in range(nodes_norm.shape[0]):
            ax.text(nodes_norm[i, 0], nodes_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]))
        for i in range(gnn_norm.shape[0]):
            bx.text(gnn_norm[i, 0], gnn_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]))
        plt.savefig('gnn/' + Dataset + Model_Name + '.png')
    # test
    best_parameter_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)
    model.load_state_dict(torch.load(best_parameter_file))
    torch.save(model.state_dict(), './classifier' + Dataset + Model_Name + '.' + str(best_epoch),
               _use_new_zipfile_serialization=False)
    if adversarial:
        Test(best_parameter_file, model, diagnosis_codes_all.cpu(), y, train_mask[:N], test_mask, Dataset)
    else:
        Test(best_parameter_file, model, X, y, train_mask, test_mask, Dataset)

    return 0


def Test(best_parameter_file, model, X, y, train_mask, test_mask, Dataset):
    print(best_parameter_file)
    model.eval()
    diagnosis_codes = copy.deepcopy(X)
    labels = y
    t_diagnosis_codes, t_labels = pad_matrix(diagnosis_codes, labels, num_category[Dataset])

    t_labels = torch.LongTensor(t_labels).cuda()

    logit, new_features, gnn_features, edges = model(torch.tensor(t_diagnosis_codes).cuda(), one_hot_labels(t_labels, 3))

    print(edges.size())

    pred = logit.argmax(dim=1)[:num_samples[Dataset]]

    accuary = accuracy_score(t_labels[train_mask].cpu().numpy(), pred[train_mask].cpu().numpy())
    precision = precision_score(t_labels[train_mask].cpu().numpy(), pred[train_mask].cpu().numpy(), average=None)
    recall = recall_score(t_labels[train_mask].cpu().numpy(), pred[train_mask].cpu().numpy(), average=None)
    f1 = f1_score(t_labels[train_mask].cpu().numpy(), pred[train_mask].cpu().numpy(), average='macro')
    print('Training data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    log_a = open(
        './Logs/'+Dataset+'/training/TEST____%s_Adam_%s.bak' % (
            Model_Name, lr), 'w+')
    print(best_parameter_file, file=log_a, flush=True)
    print('Training data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)

    accuary = accuracy_score(t_labels[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy())
    precision = precision_score(t_labels[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy(), average=None)
    recall = recall_score(t_labels[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy(), average=None)
    f1 = f1_score(t_labels[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy(), average='macro')
    print('Testing data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    print(best_parameter_file, file=log_a, flush=True)
    print('Testing data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)


lr = args.lr
algorithm = args.algo
Dataset = args.dataset
modeltype = args.model
n_lables = 3
n_epoch = 20
if modeltype == 'linear':
    linear = True
    Model_Name = 'gnn'
else:
    linear = False
    Model_Name = 'gnnlstm'
if algorithm == 'adv':
    adversarial = True
    Model_Name += 'adv'
else:
    adversarial = False

print('lr =' + str(lr))

Training(Dataset, n_epoch, lr)
