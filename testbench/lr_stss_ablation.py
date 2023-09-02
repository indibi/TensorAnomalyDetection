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
parser.add_argument('--name', metavar='Experiment name', type=str, default='lr_stss_ablation')
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

# Define experiment function
def run_exp(inputs):
    G = inputs['G']
    A = nx.adjacency_matrix(G)
    Deg = np.diag(np.asarray(np.sum(A,axis=1)).ravel())
    Dsq = np.linalg.inv(np.sqrt(Deg))
    An = Dsq@A@Dsq
    Y = inputs['Y']
    X = inputs['X']
    an_m = inputs['an_m']
    psi = inputs['psi']
    res = lr_stss(Y, An, 2,1, verbose=0, max_it2=40, max_it=300,
        lda_2=inputs['lda_2'], lda_1=inputs['lda_1'], lda_t=inputs['lda_t'],
        lda_l=inputs['lda_l'], psis=[psi,psi,psi,psi])
    
    result = {'auc': roc_auc_score(an_m.ravel(),np.abs(res['S']).ravel()),
              'rec_err': np.linalg.norm(res['X']-X)/np.linalg.norm(X),
              'graph_type': args.gt,
              'graph_seed': inputs['seed'],
              'lda_1': inputs['lda_1'],
              'lda_2': inputs['lda_2'],
              'lda_l': inputs['lda_l'],
              'lda_t': inputs['lda_t'],
              'psi_1': inputs['psi'], 'psi_2': inputs['psi'],
              'psi_3': inputs['psi'], 'psi_4': inputs['psi'],
              'maxit': args.maxit, 'maxit2': args.maxit2,
              'it': res['it']}
    
# Initialize the clients
# client = Client(n_workers=args.NoC)

with pd.HDFStore(name+'.hdf5') as f:
    f.put