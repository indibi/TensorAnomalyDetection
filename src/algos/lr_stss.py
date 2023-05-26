import numpy as np
from src.util.soft_hosvd import soft_moden
from src.util.t2m import t2m
from src.util.m2t import m2t
from src.util.soft_treshold import soft_treshold
from src.util.soft_hosvd import soft_moden
from scipy.sparse import csr_matrix

def lr_stss(Y, A, temp_m, spat_m, **kwargs):
    """Low-rank separation algorithm from spatio-temporally smooth sparse.

    WRITE ELABORATE DESCRIPTION

    Args:
        Y (np.ndarray): Observed data/signal/tensor
        A (np.array): Adjecancy matrix of the neighbour locations
        temp_m (int): temporal mode of the tensor
        spat_m (int, list of ints): spatial mode(s) of the tensor
        lda_1 (float): Hyperparameter of the sparsity. Defaults to 1
        lda_2 (float): Fidelity term that regularizes deviation from original observation
        lda_loc (float): Hyperparameter of the local smoothness. Defaults to 1
        lda_t (float): Hyperparameter of the temporal smoothness. Defaults to 1
        psis (float, list of floats): Hyperparameters/weights of the nuclear norms. Defaults to 1
        verbose (int): Algorithm verbisity level. Defaults to 1.
        max_it (int): Maximum number of iterations allowed for the algorithm. Defaults to 250
        rho_upd (float): Step size update coefficient of the ADMM algorithm. Defaults to 1.2
        rho_mu (float): Step size update treshold of the ADMM algorithm. Defaults to 10
        err_tol (float): Convergence criteria for the algorithm. Defaults to 1e-5
    """
    sz = Y.shape
    lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(sz))])
    N = len(lr_modes)
    lda_loc = kwargs.get('lda_loc',1)
    lda_1 = kwargs.get('lda1',1)
    lda_2 = kwargs.get('lda2',100)
    psis = kwargs.get('psis', tuple([1 for _ in range(N)]))
    if isinstance(psis,float):
        psis = tuple([psis]*N)
    
    verbose = kwargs.get('verbose',1)
    max_it = kwargs.get('max_it',250)
    rho = kwargs.get('rho',1)
    rho_upd = kwargs.get('rho_upd', 1.2)
    rho_mu = kwargs.get('rho_mu', 10)
    err_tol = kwargs.get('err_tol',1e-5)

    # Initialize variables
    X_i = [np.zeros(sz) for _ in range(N)]
    W_t = np.zeros(sz)
    W_loc = np.zeros(sz)
    W = np.zeros(sz)
    S = np.zeros(sz)
    X = np.zeros(sz)
    obs_idx = ~Y.mask
    unobs_idx = Y.mask

    # Initialize dual variables
    Lda_i = [np.zeros(sz) for _ in range(N)]    # X = X_i
    Lda_t = np.zeros(sz)                        # W_t = D xt S
    Lda_loc = np.zeros(sz)                      # W_loc = (I-A)xl S
    Lda = np.zeros(sz)                          # W = S

    # Initialize Delt matrix
    Delt = np.eye(sz[temp_m-1]) + np.diag(-np.ones(sz[temp_m-1]-1), 1)
    Delt[-1,0]=-1
    Delt = csr_matrix(Delt)
    I_A = csr_matrix(np.eye(A.shape[0])-A)

    it = 0
    obj = [np.inf]
    r = []  # Norm of primal residual in each iteration
    s = []  # Norm of dual residual
    rhos = [rho] # Step sizes
    while it<max_it:
        # {X, Wt, Wl, W} Block updates
        tempX = rho*sum(X_i)-sum(Lda_i)/rho
        X[obs_idx] = (lda_2*(Y[obs_idx]-S[obs_idx]) + tempX[obs_idx])/(lda_2+N*rho)
        X[unobs_idx] = tempX[unobs_idx]/(N*rho) # X update
        ## Wt update
        W_t = soft_treshold( m2t(Delt@t2m(S,temp_m),sz,temp_m)-Lda_t/rho,
                             lda_t/rho)
        ## Wl update
        W_loc = soft_treshold( m2t((I_A)@t2m(S,spat_m),sz,spat_m)-Lda_loc/rho,
                              lda_loc/rho)
        ## W update
        W = soft_treshold( S-Lda/rho,
                          lda_1/rho)
        
        # {S, Xi} Block updates
        ## Solve for S using APGD or CG
        # solve s
        b = lda_2*M*(Y-X)+ m2t(Delt.T@t2m(S,temp_m),sz,temp_m)\
                        + m2t((I_A).T@t2m(S,spat_m),sz,spat_m)
        Q = lda_2*np.diag(M.ravel()) + rho*
        S.flat = solve_qp()

        ## Update Xi
        for i,m in enumerate(lr_modes):
            X_i[i] = soft_moden(X+Lda_i[i]/rho,
                                m)            

        # Dual variable updates
        for i in range(N):
            Lda_i[i] = Lda_i[i] + rho*(X-X_i[i])
        Lda_t += rho*(W_t- m2t(Delt@t2m(S,temp_m),sz,temp_m))
        Lda_loc += rho*m2t((I_A)@t2m(S,spat_m),sz,spat_m)
        Lda += rho*(W-S)

        it +=1
        # Check convergence
        eps_pri = 
        eps_dual =
        if verbose:
            print(f"It-{it}:\t## |r|={r[-1]:.5f} \t ## |s|={s[-1]:.5f} \t ## rho={rho:.4f} obj={obj[-1]:.4f} \t ## del_obj = {obj[-1]-obj[-2]:.4f} ")
        
        if r[-1]<eps_pri and s[-1]<eps_dual:
            if verbose:
                print("Converged!")
            break
        else: # Update step size if needed
            if rho_upd !=-1:
                if r[-1]>rho_mu*s[-1]:
                    rho=rho*rho_upd
                    rhos.append(rho)
                    for i,m in enumerate(modes):
                        Inv[i] = np.linalg.inv(alpha[i]*Ls[i] + rho*np.eye(n[m-1]))
                elif s[-1]>rho_mu*r[-1]:
                    rho=rho/rho_upd
                    rhos.append(rho)
                    for i,m in enumerate(modes):
                        Inv[i] = np.linalg.inv(alpha[i]*Ls[i] + rho*np.eye(n[m-1]))
                else:
                    rhos.append(rho)

    results = {'X':X,
                'S':S,
                'obj':np.array(obj),
                'r':np.array(r),
                's':np.array(s),
                'it':it,
                'rho':np.array(rhos)}
    return results


def solve_qp():
    """_summary_
    """
    



def plot_alg(r,s,obj, rhos):
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(r)
    axs[0,0].set_xlabel("k")
    axs[0,0].set_ylabel("||r||")
    
    axs[0,1].plot(s)
    axs[0,1].set_xlabel("k")
    axs[0,1].set_ylabel("||s||")

    axs[1,0].plot(obj[1:])
    axs[1,0].set_xlabel("k")
    axs[1,0].set_ylabel("Objective")

    axs[1,1].plot(rhos)
    axs[1,1].set_xlabel("k")
    axs[1,1].set_ylabel("rho")