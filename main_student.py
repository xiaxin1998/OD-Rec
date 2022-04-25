import os
import time
import torch
import argparse
import numpy as np
from student import *
from utils import *
import pickle
from sas import SAS


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall')
parser.add_argument('--act', default='gelu')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_layer', default=1, type=int)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=128, type=int)
parser.add_argument('--inner_units', default=128, type=int)
parser.add_argument('--num_blocks', default=3, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--beta', type=float, default=0.2, help='the scale of cl')
parser.add_argument('--alpha', type=float, default=0.001, help='the scale of prediction')
parser.add_argument('--para', type=float, default=0.8, help='the scale of kd')
parser.add_argument('--tt_rank', type=float, default=60, help='8/16/32')
parser.add_argument('--b_num', type=float, default=2, help='1/2/3/4')
parser.add_argument('--STT', type=bool, default=True)
parser.add_argument('--t', type=int, default=2)
# blocks = {'2': [[169, 241], [16, 8]], '3': [[13, 13, 241], [8,4,4]]} #tmall
blocks = {'2': [[117, 316], [16, 16]], '3': [[18, 26, 79], [8,8,4]]} #rr
parser.add_argument('--blocks', type=float, default=[[169, 241], [16, 8]], help='8/16/32')
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    if opt.dataset == 'Tmall':
        n_node = 40727+2
    elif opt.dataset == 'retailrocket':
        n_node = 36968 + 4
    else:
        n_node = 309 + 1
    train_data = Data(train_data, train_data, shuffle=True, n_node=n_node, train=1)
    test_data = Data(test_data, train_data, shuffle=False, n_node=n_node, train=0)
    model = SASRec(n_node, opt)
    model = trans_to_cuda(model)
    path_state_dict = "../sas_teacher_tmall_nn.pkl"
    state = torch.load(path_state_dict)
    teacher = SAS(n_node, opt)
    teacher.load_state_dict(state)

    model_size = n_node * opt.hidden_units
    if opt.b_num == 2:
        compressed = opt.blocks[0][0] * opt.blocks[1][0] * opt.tt_rank + int((opt.blocks[0][1] * opt.blocks[1][1] * opt.tt_rank)/(opt.t*opt.t))
        print('compression rate:', float(model_size)/compressed)
    if opt.b_num == 3:
        compressed = opt.blocks[0][0] * opt.blocks[1][0] * opt.tt_rank + int((opt.blocks[0][1] * opt.blocks[1][1] * opt.tt_rank * opt.tt_rank)/(opt.t*opt.t)) + int((opt.blocks[0][2] * opt.blocks[1][2] * opt.tt_rank)/(opt.t*opt.t))
        print('compression rate:', float(model_size) / compressed)


    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]
    for epoch in range(200):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, epoch, opt, trans_to_cuda(teacher))

        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == '__main__':
    main()
