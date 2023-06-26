import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from dask.distributed import Client
from sklearn.metrics import roc_auc_score
import project_path

from src.util.generate_lr_data import generate_low_rank_data
from src.algos.lr_stss import lr_stss, plot_alg
from src.util.generate_anomaly import generate_spatio_temporal_anomaly

parser = argparse.ArgumentParser(prog='lr-stss hyperparameter ablation study',
                                description='Low rank // Spatio-temporal sparse separation'+
                                'hyperparameter effects study'
                                )

parser.add_argument('-l','--NoL', metavar='Number of Locations', type=int, default=40)
parser.add_argument('-r','--rank', metavar='Tensor rank (rank,rank,)', type=int, default=3)
parser.add_argument('--NoG', metavar='Number of graphs', type=int, default=1)
parser.add_argument('--NoT', metavar='Number of trials', type=int, default=10)
parser.add_argument('--gt', metavar='Graph Type', choices=['er','ba', 'grid', 'geometric'],
                    default='er')
parser.add_argument('--NoC', metavar='Number of cores', type=int, default=8)
parser.add_argument('--maxit', metavar='Maximum Iteration', type=int, default=300)
parser.add_argument('--maxit2', metavar='Maximum Iteration', type=int, default=40)
parser.add_argument('--amp', metavar='Noise magnitude', default=1)
args=parser.parse_args()

print(args)
# Initialize the clients
# client = Client(args.NoC)