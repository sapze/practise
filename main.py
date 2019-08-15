# -*- coding: utf-8 -*-
import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--g_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--d_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--lambda_gen', type=int, default=20, help='the number of epochs')
parser.add_argument('--lambda_dis', type=int, default=20, help='the number of epochs')
parser.add_argument('--lr_gen', type=float, default=0.02, help='learning rate of RS task')
parser.add_argument('--lr_dis', type=float, default=0.01, help='learning rate of KGE task')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')



show_loss = False
show_topk = False

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)
