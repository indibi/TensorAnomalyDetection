function [L, S, times] = cp_based(Y, param)
% [L, S, Nt] = cp_based(Y, param)
% CP Based Low rank plus Sparse Decomposition

N = ndims(Y);
sz = size(Y);
mask_Y = ones(sz)>0;
mask_Y(param.ind_m) = ~mask_Y(param.ind_m);
max_iter = param.max_iter;
err_tol = param.err_tol;
mu = param.mu;
R = param.init_rank;
U = cell(1,N);
for n=1:N
    U{n} = randn(sz(n),R);
end
lambda = param.lambda;
beta_1 = param.beta_1;
S = zeros(size(Y));
L = S;
Lam{1} = zeros(size(Y));

times = [];
iter = 1;
obj_val = compute_obj(Y,L,S,U,Lam,param);
while true
    %% L, Fac Update
    tstart = tic;
    [L, U] = cp_fac_update_als(Y-S-Lam{1}, U, param.ind_m, mu);
    times(iter,1) = toc(tstart);
    
    %% S Update
    tstart = tic;
    temp = Y-L-Lam{1};
    Sold = S;
    S(mask_Y) = soft_threshold(temp(mask_Y), lambda);
    times(iter,2) = toc(tstart);
    
    %% Dual Updates
    tstart = tic;
    Lam{1} = Lam{1} + Y-L-S;
    times(iter,3) = toc(tstart);
    
    %% Error and objective calculations
    obj_val(iter+1) = compute_obj(Y,L,S,U,Lam,param);
    err = norm(S(:)-Sold(:))/norm(Sold(:));
    iter = iter+1;
    
    if err<=err_tol
        disp('Converged!')
        break;
    end
    if iter>max_iter
        disp('Max iter')
        break;
    end
end
% temp = zeros(size(Y));
% for i=1:N
%     temp = temp+Lx{i};
% end
% Lx = temp/N;

end

function [val, term] = compute_obj(Y,L,S,U,Lam,param)

N = ndims(Y);
lambda = param.lambda;
mu = param.mu;
beta_1 = param.beta_1;
term = zeros(1,3);

term(1) = sum((Y-S-L-Lam{1}).^2,'all');
for i=1:N
    term(2) = term(2) + mu/2*norm(U{i},'fro');
end
term(3) = lambda*sum(abs(S),'all');
val = sum(term);
end