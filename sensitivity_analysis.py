import argparse
from utils import *
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='sensitivity')  # 创建parser对象
parser.add_argument('--dataset', default='Splice', type=str, help='dataset')
args = parser.parse_args()  # 解析参数，此处args是一个命名空间列表
if args.dataset == 'Splice':
    from Splicemodels import *
else:
    from IPSmodels import *


def sensitivity_analysis(Model_Name, Dataset):
    n_labels = num_classes[Dataset]
    batch_size = 128
    if 'gnn' in Model_Name:
        batch_size = num_samples[Dataset]
    best_parameters_file = model_file(Dataset, Model_Name)
    if 'gnnlstm' in Model_Name:
        model = GNNLSTMtest(Model_Name)
    elif 'gnn' in Model_Name:
        model = GNNtest(Model_Name)
    elif Dataset == 'Splice':
        model = SpliceLSTM()
    else:
        model = IPSLSTM()
    if torch.cuda.is_available():
        model = model.cuda()
    X, y = load_data(Dataset, False)
    model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
    model.eval()
    print('-----------attack--------------')
    n_batches = int(np.ceil(float(len(X)) / float(batch_size)))
    index_all = np.array([])
    for index in range(n_batches):  # n_batches

        batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
        batch_labels = y[batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, num_category[Dataset])
        logit = model(torch.tensor(t_diagnosis_codes).cuda())
        logit = logit.data.cpu().numpy()
        y_pred = np.argmax(logit, 1)
        correct_index = np.where(batch_labels == y_pred)[0]
        correct_index += batch_size * index
        index_all = np.concatenate((index_all, correct_index))

    index_all = np.array(index_all, dtype=np.int32)
    n_batches = int(np.ceil(float(len(index_all)) / float(batch_size)))
    X = X[index_all]
    y = y[index_all]
    feature_sensitivity = np.zeros((num_feature[Dataset], num_avail_category[Dataset], len(y)))
    for index in range(n_batches):  # n_batches

        batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
        batch_labels = y[batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, num_category[Dataset])
        logit = model(torch.tensor(t_diagnosis_codes).cuda())
        logit = logit.data.cpu().numpy()
        t_labels = torch.LongTensor(t_labels)
        y_para = one_hot_labels(t_labels, n_labels).cpu().numpy()
        y_prob = logit * y_para
        y_prob = np.max(y_prob, axis=1)

        for i in range(num_feature[Dataset]):
            batch_diagnosis_codes_temp = copy.deepcopy(batch_diagnosis_codes)
            for j in range(num_avail_category[Dataset]):
                for k in range(len(batch_diagnosis_codes_temp)):
                    batch_diagnosis_codes_temp[k][i] = j
                t_diagnosis_codes_temp, t_labels_temp = pad_matrix(batch_diagnosis_codes_temp, batch_labels, num_category[Dataset])
                logit = model(torch.tensor(t_diagnosis_codes_temp).cuda())
                logit = logit.data.cpu().numpy()
                y_prob_temp = logit * y_para
                y_prob_temp = np.max(y_prob_temp, 1)
                y_prob_diff = y_prob - y_prob_temp
                feature_sensitivity[i][j][batch_size * index: batch_size * (index + 1)] = y_prob_diff

    feature_sensitivity_max = np.max(feature_sensitivity, 1)
    mean_sensitivity = np.mean(feature_sensitivity_max, 1)
    pickle.dump(mean_sensitivity, open(
        './Logs/'+Dataset+'/'+Model_Name+'/sensitivity.pickle', 'wb'))
    return mean_sensitivity


def plot_seperate(Model_Name, Dataset):
    x = range(num_feature[Dataset])
    y = sensitivity_analysis(Model_Name, Dataset)
    plt.title(Dataset + ',' + Model_Name)
    plt.xlabel("Features")
    plt.ylabel("Mean Prediction Changes")
    plt.bar(x, y, width=0.8)
    plt.xlim((0, num_feature[Dataset]))
    plt.savefig('./sensitivity/'+Dataset+'_'+Model_Name+'.png')
    plt.show()

dataset = args.dataset
for modelname in Model[dataset].keys():
    plot_seperate(modelname, dataset)