import numpy as np
from src.util.t2m import t2m
from src.util.m2t import m2t

def loss(B, Ls, **kwargs):
    """Low-rank component, Sparse and locally smooth component seperation.

    Args:
        B (np.ndarray): Observed data
        Ls (np.ndarray): Graph laplacians
    Returns:
        X (np.ndarray): Seperated low-rank and smooth part
        S (np.ndarray): Seperated sparse part 
    """
    pass
