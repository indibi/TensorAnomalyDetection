import numpy as np

def soft_treshold(x, tau):
    """Soft tresholds any numpy array with element-wise with the treshold tau and returns it.
    Args:
        x (np.array): Vector/matrix/array to be tresholded
        tau (float): Treshold

    Returns:
        y: Tresholded array
    """
    if tau <0:
        raise ValueError("The threshold value tau is negative")
    y = x.copy().astype('float64')
    y[x>tau] = x[x>tau] -tau
    y[x<-tau] = x[x<-tau] + tau
    y[(-tau<=x) & (x<=tau)]=0
    return y