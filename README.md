# YujunZhou-cs326

## Training a classifier

### parameters:
lr: learning rate, float, default = 0.0001

dataset: 'IPS' or 'Splice', 'str', default = 'Splice'

For LSTM_training.py (baseline training):

    adv: Adversarial training or normal training, bool,  default = False
    
For GNN_training.py (our model training):

    algo: Adversarial training or normal training, 'str' ,  default = 'Normal'. The other choice is 'adv'.
    
    model: GNN+linear or GNN+LSTM, 'str', default = 'linear'. The other choice is 'LSTM'
    
### Commands:
python LSTM_training.py --dataset IPS --lr 0.0002 --adv True

python GNN_training.py

## Robustness Assessment

### parameters
budget: attack budget, the largest number of features the attack can modify, 'int', default = 5

dataset: 'IPS' or 'Splice', 'str', default = 'Splice'

modeltype: 'Normal', 'adversarial', 'gnn', 'gnnadv', 'gnnlstm', 'gnnlstmadv', 'str', default = 'Normal'

time: time limit, 'int', default = 60 (for FSGS on IPS, it is set to 300)

### Commands:
python FSGSmax.py --dataset IPS --budget 5  --modeltype gnnlstm --time 300

python OMPGSmax.py

## Visualization

### parameters:
dataset: 'IPS' or 'Splice', 'str', default = 'Splice'

### Commands:
python visualization.py --dataset IPS

## Best performance running:
python visualization.py --dataset IPS

python visualization.py --dataset Splice

python FSGSmax.py --dataset IPS --budget 5  --modeltype gnnlstmadv --time 300

python OMPGSmax.py --dataset IPS --budget 5  --modeltype gnnlstmadv --time 60

python FSGSmax.py --dataset Splice --budget 5  --modeltype gnnlstmadv --time 120

python OMPGSmax.py --dataset Splice --budget 5  --modeltype gnnlstmadv --time 60

python FSGSmax.py --dataset IPS --budget 5  --modeltype adversarial --time 300

python OMPGSmax.py --dataset IPS --budget 5  --modeltype adversarial --time 60

python FSGSmax.py --dataset Splice --budget 5  --modeltype adversarial --time 60

python OMPGSmax.py --dataset Splice --budget 5  --modeltype adversarial --time 60
