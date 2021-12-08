import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from utils import *


# creating parser object
parser = argparse.ArgumentParser(description='gene')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--adv', default=False, type=bool, help='adversarial training or not')
parser.add_argument('--dataset', default='Splice', type=str, help='Dataset')
args = parser.parse_args()

# There are two datasets, some models have the same name for the two dataset, so we just load one depending on the detaset.
if args.dataset == 'Splice':
    from Splicemodels import *
else:
    from IPSmodels import *


# used to calculate the loss of the validation set
def calculate_cost(model, X, y, batch_size):
    n_batches = int(np.ceil(float(len(X)) / float(batch_size)))
    cost_sum = 0.0
    for index in range(n_batches):
        batch_diagnosis_codes = X[batch_size * index: batch_size * (index + 1)]
        batch_labels = y[batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, num_category[Dataset])
        t_labels = torch.LongTensor(t_labels).cuda()
        logit = model(torch.tensor(t_diagnosis_codes).cuda())
        loss = F.cross_entropy(logit, t_labels)
        cost_sum += loss.cpu().data.numpy()
    return cost_sum / n_batches


def Training(Dataset, batch_size, n_epoch, lr):
    # devide the dataset into train, validation and test
    train_idx, test_idx = train_test_split(np.array(range(num_samples[Dataset])), test_size=0.1, random_state=666)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=66)

    X, y = preparation(Dataset)
    # in adversarial training, the training data is the adversarial samples, or the samples are clean samples
    if adversarial:
        if Dataset == 'Splice':
            modified_funccall_file = open('./Logs/'+Dataset+'/Normal/greedmax_modified_funccall_5.pickle', 'rb')
            modified_funccall_label = pickle.load(open('./dataset/SpliceY.pickle', 'rb'))
        else:
            modified_funccall_file = open('./Logs/' + Dataset + '/Normal/gradmax_modified_funccall_5.pickle', 'rb')
            modified_funccall_label = pickle.load(open('./dataset/IPSY.pickle', 'rb'))
        modified_funccall = pickle.load(modified_funccall_file)

        y_Train = modified_funccall_label
        X_Train = modified_funccall
    else:
        y_Train = y[train_idx]
        X_Train = X[train_idx]

    y_Test = y[test_idx]
    X_Test = X[test_idx]

    y_Validation = y[val_idx]
    X_Validation = X[val_idx]

    output_file = './outputs/'+Dataset+'/' + Model_Name + '/' + str(lr) + '/'
    make_dir('./outputs/')
    make_dir('./outputs/'+Dataset+'/')
    make_dir('./outputs/'+Dataset+'/' + Model_Name + '/')
    make_dir('./outputs/'+Dataset+'/' + Model_Name + '/' + str(lr) + '/')
    make_dir('./Logs/'+Dataset+'/training/')

    log_f = open(
        './Logs/'+Dataset+'/training/TEST_%s_%s.bak' % (
            Model_Name, lr), 'w+')
    print('constructing the optimizer ...', file=log_f, flush=True)

    if Dataset == 'Splice':
        model = SpliceLSTM()
    else:
        model = IPSLSTM()
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('done!', file=log_f, flush=True)
    # define cross entropy loss function
    CEloss = torch.nn.CrossEntropyLoss().cuda()
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    print('training start', file=log_f, flush=True)
    model.train()

    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    epoch_duaration = 0.0
    best_epoch = 0.0

    for epoch in range(n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)

        # start training with randomly input batches.
        for index in samples:
            # make X like one hot vectors.
            batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
            batch_labels = y_Train[batch_size * index: batch_size * (index + 1)]
            t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, num_category[Dataset])
            t_labels = torch.LongTensor(t_labels).cuda()

            optimizer.zero_grad()

            logit = model(torch.tensor(t_diagnosis_codes).cuda())
            loss = CEloss(logit, t_labels)
            loss.backward()

            optimizer.step()
            cost_vector.append(loss.cpu().data.numpy())

            iteration += 1

        duration = time.time() - start_time
        train_cost = np.mean(cost_vector)
        validate_cost = calculate_cost(model, X_Validation, y_Validation, batch_size)
        epoch_duaration += duration

        # if the current validation cost is smaller than the current best one, then save the model
        if validate_cost < best_validate_cost:
            # for some pytorch edition, the following function do not work, '_use_new_zipfile_serialization=False'
            # should be used

            # torch.save(rnn.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch))
            torch.save(model.state_dict(), output_file + Dataset + Model_Name + '.' + str(epoch),
                       _use_new_zipfile_serialization=False)
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration), file=log_f, flush=True)

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_epoch = epoch

        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f' % (best_epoch, best_train_cost, best_validate_cost)
        print(buf, file=log_f, flush=True)
        print()

    # test

    print('-----------test--------------', file=log_f, flush=True)
    best_parameters_file = output_file + Dataset + Model_Name + '.' + str(best_epoch)


    print(best_parameters_file)
    model.load_state_dict(torch.load(best_parameters_file))
    torch.save(model.state_dict(), './classifier' + Dataset + Model_Name + '.' + str(best_epoch),
               _use_new_zipfile_serialization=False)
    model.eval()
    n_batches = int(np.ceil(float(len(X_Train)) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])

    # test for the training set
    for index in range(n_batches):  # n_batches

        batch_diagnosis_codes = X_Train[batch_size * index: batch_size * (index + 1)]
        batch_labels = y_Train[batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, num_category[Dataset])

        logit = model(torch.tensor(t_diagnosis_codes).cuda())
        prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()
        y_true = np.concatenate((y_true, t_labels))
        y_pred = np.concatenate((y_pred, prediction))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Training data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    log_a = open(
        './Logs/'+Dataset+'/training/TEST____%s_Adam_%s.bak' % (Model_Name, lr), 'w+')
    print(best_parameters_file, file=log_a, flush=True)
    print('Training data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)

    # test for test set
    y_true = np.array([])
    y_pred = np.array([])
    n_batches_test = int(np.ceil(float(len(X_Test)) / float(batch_size)))
    for index in range(n_batches_test):  # n_batches

        batch_diagnosis_codes = X_Test[batch_size * index: batch_size * (index + 1)]
        batch_labels = y_Test[batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, num_category[Dataset])

        logit = model(torch.tensor(t_diagnosis_codes).cuda())
        prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()
        y_true = np.concatenate((y_true, t_labels))
        y_pred = np.concatenate((y_pred, prediction))

    accuary = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Testing data')
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1))

    print('Testing data', file=log_a, flush=True)
    print('accuary:, precision:, recall:, f1:', (accuary, precision, recall, f1), file=log_a, flush=True)


lr = args.lr
adversarial = args.adv
Dataset = args.dataset
if adversarial:
    Model_Name = 'LSTM'
else:
    Model_Name = 'LSTMadv'
batch_size = 32
n_epoch = 10000
n_lables = 3

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

Training(Dataset, batch_size, n_epoch, lr)

