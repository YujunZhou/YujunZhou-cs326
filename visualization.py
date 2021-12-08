from utils import *
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='sensitivity')  # 创建parser对象
parser.add_argument('--dataset', default='IPS', type=str, help='dataset')
args = parser.parse_args()  # 解析参数，此处args是一个命名空间列表
if args.dataset == 'Splice':
    from Splicemodels import *
else:
    from IPSmodels import *


# visualize all the models of the dataset
def visualize(Dataset):
    tsne = TSNE(n_components=2)
    fig = plt.figure(figsize=(14, 15))
    fignum = 1
    X, y = load_data(Dataset, False)
    for model in Model[Dataset]:
        gnn_features_cpu = pickle.load(open('./gnn/'+Dataset+model+'.gnnout.pickle', 'rb')).cpu().data.numpy()
        gnn_tsne = tsne.fit_transform(gnn_features_cpu, y)
        gnn_min, gnn_max = gnn_tsne.min(0), gnn_tsne.max(0)
        gnn_norm = (gnn_tsne - gnn_min) / (gnn_max - gnn_min)
        ax = fig.add_subplot(3, 2, fignum)
        plt.setp(ax, xticks=[], yticks=[])
        fignum += 1
        ax.set_title(Dataset+'_'+model, fontsize=25)
        for i in range(gnn_norm.shape[0]):
            ax.text(gnn_norm[i, 0], gnn_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]))
    plt.savefig('gnn/' + Dataset + '.png')


# get the final feature of LSTM based classifiers
def get_feature(Dataset, Model_Name):
    if Dataset == 'Splice':
        model = SpliceLSTM_temp()
    else:
        model = IPSLSTM_temp()
    if torch.cuda.is_available():
        model = model.cuda()
    best_parameters_file = model_file(Dataset, Model_Name)
    model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
    model.eval()
    X, y = load_data(Dataset, False)
    batch_size = 8
    n_batches = int(np.ceil(float(len(X)) / float(batch_size)))
    feature = torch.tensor([]).cuda()
    for index in range(n_batches):  # n_batches
        batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
        batch_labels = y
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, num_category[Dataset])
        t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()

        logit = model(t_diagnosis_codes)
        feature = torch.cat((feature, logit))
    feature = feature.cpu()
    pickle.dump(feature, open('./gnn/' + Dataset + Model_Name + '.gnnout.pickle', 'wb'))


visualize(args.dataset)