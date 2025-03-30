import numpy as np
import pandas as pd

def t2m(X, m=1):
    """Matricisez the tensor X in the m'th mode.
    
    It is done by stacking fibers of mode m as column vectors.
    Order of the other modes follow cyclic order.
    ie ( I_m x I_(m+1). ... .I_N x I_0. ... I_(m-1) ).
    Args:
        X (np.ndarray): Tensor to be matricized
        # m (int, optional): The mode whose fibers are stacked as vectors. Defaults to 1.
    Returns:
        M (np.ndarray): Matricized tensor.
    """
    n = X.shape
    if m>len(n) or m<1:
        raise ValueError(f"Invalid unfolding mode provided. m={m}, X shape:{n}")
    Xm = __roll_2_dim(X,m).ravel().reshape((n[m-1], int(np.prod(n)/n[m-1])))
    return Xm

def __roll_2_dim(X, m):
    n = X.shape; N = len(n)
    dest = np.arange(N)
    src = (np.arange(N) + (m-1))%N
    if isinstance(X, np.ndarray):
        return np.moveaxis(X, src, dest)
    elif isinstance(X, torch.Tensor):
        return torch.moveaxis(X, tuple(src), tuple(dest))

def convert_index(n,i,k):
    """Convert the index of an element of a tensor into it's k'th mode unfolding index

    Args:
        n (tuple): Tensor shape
        i (tuple): Tensor index
        k (int): Mode unfolding
    Returns:
        idx (tuple): Index of the element corresponding to the matricized tensor
    """
    if type(n) != tuple:
        raise TypeError("Dimension of the tensor, n is not a tuple")
    if type(i) != tuple:
        raise TypeError("index of the tensor element, i is not a tuple")
    if len(n) ==1:
        raise ValueError(f"The provided dimension n={n} is for the vector case")
    for j, i_ in enumerate(i):
        if i_>= n[j]:
            raise ValueError(f"Index i exceeds the dimension n in {j+1}'th mode")
    if k > len(n) or k<1:
        raise ValueError(f"Unfolding mode {k} is impossible for {len(n)}'th order tensor.")
    j=0
    idx = [i[k-1]]
    n_ = list(n)
    n_ = n_[k:] + n_[:k-1]
    i_ = list(i)
    i_ = i_[k:] + i_[:k-1]
    for p,i_k in enumerate(i_):
        j+= i_k*np.prod(n_[p+1:])
    idx.append(int(j))
    return tuple(idx)


def m2t(Xm, dims, m=1):
    """Tensorizes the matrix obtained by t2m to its original state.
# 
    Args:
        Xm (np.ndarray): Matrix
        dims (tuple): original dimensions of the tensor
        m (int): The mode for which the matricization was originally made with t2m. Defaults to 1.
    Returns:  
        X (np.ndarray): Tensor
    """
    N = len(dims); 
    if m > N or m <1:
        raise ValueError(f"Invalid tensorization order m={m}, N={N}")
    
    old_dest = (np.arange(N) + (m-1))%N
    dims2 = tuple([dims[i] for i in old_dest])
    X = Xm.ravel().reshape(dims2)
    X = __unroll_from_dim(X, m)
    return X

def __unroll_from_dim(X, m):
    n = X.shape
    N = len(n)
    dest = (np.arange(N) + (m-1))%N
    src = np.arange(N)
    if isinstance(X, np.ndarray):
        return np.moveaxis(X, src, dest)
    elif isinstance(X, torch.Tensor):
        return torch.moveaxis(X, tuple(src), tuple(dest))



def likelihood_score(A,S,rad,h=30): 
    """ A: Ajjacency matrix
        S: Estimated Sparse tensor
        h: kernel parameter
        rad: radius
        local_mode = 1
        time_mode = 2
    """
    S_loc = t2m(S, m = 1)
    loc = S.shape[0]
    time = S.shape[1]
    col = S_loc.shape[1]
    block = col/time
    r_A = np.eye(S_loc.shape[0])+np.linalg.matrix_power(A.toarray(),rad)
    likelihood = np.zeros(S_loc.shape)
    
    for s in range(loc):
        mask = r_A[np.where(r_A[s,]!=0),:].astype(bool)
        nbd = S_loc[mask[:,0,:].ravel(),:]
        # Append neighbors from additional columns in mask
        for m in range(1, mask.shape[1]):
            nbd = np.vstack((nbd,S_loc[mask[:,m,:].ravel(),:] ))
            
        W = np.zeros(nbd.shape)
        # Iterate through the columns in steps of block_size
        for i1 in range(0, col, block):
            # Slice the matrix to get the block (columns from i to i + block_size)
            if i1==0:
                block = nbd[:, i1:i1 + block]
                for loc in range(nbd.shape[0]):
                    W[loc, i1:i1+block] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1:i1+block]),0,h)
            elif i1==col-block:
                block = nbd[:, i1-block:i1 + block]
                for loc in range(nbd.shape[0]):
                    W[loc, i1:i1+block] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1-block:i1+block]),0,h)
            else:
                block = nbd[:, i1-block:i1 + (2*block)]
                for loc in range(nbd.shape[0]):
                    W[loc, i1:i1+block] = norm.pdf(np.linalg.norm(block[loc,:] - S_loc[s,i1-block:i1+(2*block)]),0,h)
                    
        mean = np.sum(W * nbd) / np.sum(W)
        sd = np.sqrt(np.sum(W * (nbd - mean)**2) / np.sum(W))
        likelihood[s,] = np.log(sd) + (0.5*np.power(((S_loc[s,] - mean)/sd),2))
        return likelihood
    