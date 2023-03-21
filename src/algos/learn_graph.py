import numpy as np
from src.util.t2m import t2m
from src.util.m2t import m2t
import enum
from numpy.linalg import inv, norm
from numpy.linalg import multi_dot as mdot


def learn_graph(Y,n,modes, rho, beta, max_it=100, eps=1e-4, alpha=None, A_inv=None, Pst=None, P=None, verbose=0):
    """ Learns a sparse graph structure from a tensor in each modes.  

    Args:
        Y (np.array): _description_
        n (_type_): _description_
        modes (_type_): _description_
        rho (_type_): _description_
        beta (_type_): _description_
        max_it (int, optional): _description_. Defaults to 100.
        eps (_type_, optional): _description_. Defaults to 1e-4.
        alpha (_type_, optional): _description_. Defaults to None.
        A_inv (_type_, optional): _description_. Defaults to None.
        Pst (_type_, optional): _description_. Defaults to None.
        P (_type_, optional): _description_. Defaults to None.
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    sz = lambda s : int((s*(s-1))/2)
    if P == None:
        P =init_P(n, modes)
    Ps = [np.sum(P[i],0) for i in range(len(P))] # 1.T @ Pi
    Pst = [Ps[i].reshape((Ps[i].size,1)) for i in range(len(Ps))] # Pi.T @ 1
    A_inv = _init_A_inv(n,modes, beta, P, Pst, rho)

    
    if alpha == None:
        alpha = [1 for _ in modes]

    omega, gamma= init_dual(n,modes)
    l,w = init_l(n,modes)
    y, dy = Y_to_y(Y, n, modes)
    
    w_old = [np.zeros((sz(n[m-1]),1))for m in modes]
    r1 = [np.zeros([sz(n[m-1])+1,1]) for m in modes] # primal residual for first ADMM block
    s1 = [ np.zeros([sz(n[m-1]),1]) for m in modes] # dual residual for second ADMM block
    r1_val =[[] for _ in modes]; s1_val=[[] for _ in modes]
    del_obj_l = np.inf;
    terms_l = [[],[],[],[]]; obj_l_val = [];

    finished_modes = [False for _ in modes]
    iter = [0 for _ in modes]  
    while min(finished_modes) <1 and max(iter)<max_it:# and del_obj_l > 1e-4:## r1_val[-1] > eps_pri and s1_val[-1] > eps_duo: 
        # convergence criteria for each mode should be seperate. CODE IS SUBOPTIMAL
        
        for i,m in enumerate(modes):    

            if ~finished_modes[i]:    
                # l updates
                l[i] = -A_inv[i]@ (alpha[i]*(2*y[i]- P[i].T@dy[i])+ 
                                    (-gamma[i]+rho[0]*n[m-1])*Pst[i]+ 
                                    omega[i] - rho[0]*w[i]
                )
                # w updates
                w_old[i] = w[i].copy()
                w[i] = l[i]+(omega[i]/rho[0])
                w[i][w[i]>0]=0
                
                # l[i] = -A_inv[i]@ (alpha[i]*(2*y[i]- P[i].T@dy[i])+ 
                #                     (-gamma[i]+rho*n[m-1])*Pst[i])

                # l[i][l[i]>0]=0
                
                s1[i] = -rho[0]*(w[i]-w_old[i])
                r1[i][:-1] = l[i]-w[i]
                r1[i][-1] = -Pst[i].T@l[i]-n[m-1]
                gamma[i]+= -rho[0]*(Pst[i].T@l[i]+n[m-1])
                omega[i]+= rho[0]*(l[i]-w[i])

                
                r1_val[i].append(0 +norm(r1[i]))
                s1_val[i].append(0 +norm(s1[i]))
                if verbose ==2:
                    print(f"It[{i}]-{iter[i]} r1_val[{i}] = {r1_val[i][-1]}, s1_val[{i}] = {s1_val[i][-1]}")
                if (r1_val[i][-1]<eps) and (s1_val[i][-1]<eps):
                    finished_modes[i] = True 
                else:
                    iter[i]+=1
            #eps_pri, eps_duo = eps_l(l,w,Pst,omega,gamma,n,modes,p_,k_, eps_rel, eps_abs)


        val, term = compute_obj_l(alpha,beta,rho,
                l,w,y,dy,P,omega,gamma,
                Y,modes)
        obj_l_val.append(val)
        #del_obj_l = abs(obj_l_val[-1]-obj_l_val[-2])
        for i in range(len(terms_l)):
            terms_l[i].append(term[i])
    return l, iter, obj_l_val, r1_val, terms_l


def compute_obj_l(alpha,beta,rho,
                l,w,y,dy,P,omega,gamma,
                Y,modes):
    n = Y.shape; M = len(modes)
    Pl = [P[i]@l[i] for i in range(M)]
    term = [
        sum([alpha[i]*(2*y[i].T@l[i] - dy[i].T@Pl[i]) for i in range(M)]),
        sum([(beta[i][0]*norm(Pl[i])**2 + beta[i][1]*norm(l[i])**2)/2 for i in range(M)]),
        sum([gamma[i]*(sum(Pl[i])+n[m-1]) + rho[0]*(sum(Pl[i]+n[m-1])**2)/2 for i,m in enumerate(modes)]),
        sum([omega[i].T@(l[i]-w[i])+rho[0]*norm(l[i]-w[i])**2/2 for i in range(M)])
    ]
    return float(sum(term)), term


def Y_to_y(Y, n, modes): # ++
    sz = lambda s : int((s*(s-1))/2) 
    M = len(modes)
    y = [np.zeros((1,sz(n[m-1]))) for m in modes] 
    dy = [np.zeros((1,sz(n[m-1]))) for m in modes]
    Yi = [t2m(Y,m) for m in modes]
    y_idx = [np.triu_indices(n[m-1],1) for m in modes]
    dy_idx = [np.diag_indices(n[m-1]) for m in modes]
    
    for i in range(M):
        tmpY = Yi[i]@Yi[i].T
        y[i] = tmpY[y_idx[i]]
        y[i] = y[i].reshape((y[i].size,1))
        dy[i] = tmpY[dy_idx[i]]
        dy[i] = dy[i].reshape((dy[i].size,1))
    return (y, dy)

def _init_A_inv(n,modes, beta, P, Pst, rho): 
    ## This needs to be tweaked and very inefficient. scipy sparse matrices could be used.       
    """ Creates the constant inverse matrices used in the graph laplacian updates.

    Args:
        n (list or tuple): Sizes of the original data dimensions
        modes (list or tuple): modes for which a graph will be estimated 
        beta (List): Laplacian diagonal hyperparameters
        P (list of np.ndarray): Row sum matrices P_i
        Pst (list of np.ndarray): P_i.T @ 1
        rho (double): ADMM step size

    Returns:
        A_inv (list of lists): 
            A_inv[i] = inv(rho_1*Pi.T@Pi ) <== l_i update matrix
    """
    sz = lambda s : int((s*(s-1))/2)
    A_inv = [None for _ in modes]
    
    for i, m in enumerate(modes): 
        # l_i update matrix: 
        # A_inv[i] =  inv(
        #     rho*(Pst[i]@Pst[i].T + np.eye(sz(n[m-1]))) + 
        #     beta[i][0]*P[i].T@P[i] + beta[i][1]*np.eye(sz(n[m-1]))
        #     )
        A_inv[i] =  inv(
                rho[0]*(Pst[i]@Pst[i].T + np.eye(sz(n[m-1]))) + 
                beta[i][0]*P[i].T@P[i] + beta[i][1]*np.eye(sz(n[m-1]))
                )        

    return A_inv


def init_dual(n, modes): 
    """Initializes the dual variables

    Args:
        sizes (list,tuple): Sizes of the original data dimensions

    Returns:
        eta: list of vectors              # Li block dual variables
            Pi@li+d_li=0
        gamma: list of doubles          
            1'@d_li-n_i=0
        omega: list of vectors 
            li-wi=0

        Lda[:-1]: list of matrices            # Yi block dual variables
            Y-Yi=0
        Lda[-1]: Tensor   
    """
    mshape = lambda n,m : (n[m-1], int(np.prod(n)/n[m-1]))
    sz = lambda s : int((s*(s-1))/2)
    gamma = [0 for _ in modes]
    omega = [np.zeros((sz(n[m-1]),1)) for m in modes]
    return (omega, gamma)

def init_l(n,modes):
    sz = lambda s : int((s*(s-1))/2)
    l = [np.zeros((sz(n[m-1]),1)) for m in modes]
    w = [np.zeros((sz(n[m-1]),1)) for m in modes]
    return (l ,w)

def init_P(n, modes): # +
    ''' Initialize the row sum matrices for mode graphs.
    '''
    sz = lambda s : int((s*(s-1))/2)
    P = [np.zeros((n[m-1], sz(n[m-1]))) for m in modes]
    for i, m in enumerate(modes):
        for j in range(n[m-1])[1:]:
            k=1; a=j-1
            while (k<=j):
                P[i][j,a]=1
                a+=n[m-1]-1-k
                k+=1
        a =0;j =0 
        while(j<n[m-1]-1):
            b= a+ (n[m-1]-1-j)
            P[i][j,a:b]=1
            a= b; j+=1
    return P

def vec_to_L(P, l):
    d_l = [-P[i]@l[i] for i in range(len(l))]
    sizes = [(len(dli),len(dli)) for dli in d_l ]
    u_indices = [np.triu_indices(s[0],1) for s in sizes]
    l_indices = [tuple(reversed(ind)) for ind in u_indices]
    l_indices = [tuple(reversed(ind)) for ind in u_indices]
    d_indices = [np.diag_indices(sizes[i][0]) for i in range(len(sizes))]
    L = []
    for i in range(len(d_l)):
        L.append(np.zeros((len(d_l[i]),len(d_l[i]))))
        L[i][u_indices[i]] = l[i].T
        L[i][l_indices[i]] = l[i].T
        L[i][d_indices[i]] = d_l[i].T
    return L


def edges_in_L(L, P, thr):
    Ln = normalize_L(L)
    E =np.zeros(Ln.shape)
    E[np.abs(Ln)>thr]=1
    idx = np.diag_indices_from(E)
    E[idx] = 0
    return E


def normalize_L(L):
    return L/norm(L)

def F_measure(b, GT_edge, E):
    G_e = [e for e in GT_edge.edges]
    G_E = np.zeros(E.shape)

    for ind in G_e:
        G_E[ind]=1
    G_E = G_E+ G_E.T 
    p_ind = E==1
    n_ind = E==0
    tp = sum(G_E[p_ind])
    fp = sum(sum(p_ind))-tp
    fn = sum(G_E[n_ind])
    tn = sum(sum(n_ind))-fn

    if tp ==0:
        P = 0; R = 0
        Fb = 0
    else :
        P = tp/(tp+fp); R = tp/(tp+fn)
        Fb = ((1+b**2)*P*R)/((b**2)*P+R )
    return (Fb, P, R)
