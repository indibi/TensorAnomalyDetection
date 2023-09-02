import numpy as np
from numpy.linalg import eigh

def bp_wavelet_kernel(x, alpha, beta, x2, x1=1):
    """Bandpass graph wavelet kernel.

        Proposed in 'D. Hammond, P. Vandergheynst, and R. Gribonval, “Wavelets on
        graphs via spectral graph theory,” Appl. Comput. Harmon. Anal., vol.
        30, no. 2, pp. 129–150, 2011'

    Args:
        L (np.array): Laplacian matrix of the graph
        alpha (int): _description_
        beta (int): _description_
        x1 (float): Lowest frequency of the frequency band
        x2 (float): Highest frequency of the frequency band
    """

        a1 = (alpha*x2 - beta*x1)/(x1*x2*(x1 - x2)^2)
        a2 =-(2*alpha*x2**2 - 2*beta*x1**2 + alpha*x1*x2 - beta*x1*x2)/(x1*x2*(x1 - x2)**2)
        a3 = (- beta*x1**3 - 2*beta*x1**2*x2 + 2*alpha*x1*x2**2 + alpha*x2**3)/(x1*x2*(x1 - x2)**2)
        a4 = (beta*x1**2 - alpha*x2**2 - 2*x1*x2 + x1**2 + x2**2)/(x1**2 - 2*x1*x2 + x2**2)
