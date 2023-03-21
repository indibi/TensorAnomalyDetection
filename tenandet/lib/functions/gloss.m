function [L, S, Nt, times] = gloss_3(Y, param)
% [L, S, Nt] = gloss_3(Y, param)
% Graph Regularized Low rank plus Smooth-Sparse Decomposition

N = ndims(Y);
sz = size(Y);
mask_Y = ones(sz)>0;
mask_Y(param.ind_m) = ~mask_Y(param.ind_m);
max_iter = param.max_iter;
err_tol = param.err_tol;
alpha = param.alpha;
theta = param.theta;
lambda = param.lambda;
gamma = param.gamma;
psi = param.psi;
beta_1 = param.beta_1;
beta_2 = param.beta_2;
beta_3 = param.beta_3;
beta_4 = param.beta_4;
beta_5 = param.beta_5;
Lx = cell(1, N);
for i=1:N
    Lx{i} = zeros(size(Y));
end
G = Lx;
S = zeros(size(Y));
L = S;
Nt = S;
W = S;
Z = S;
for i=1:N
    mods = setdiff(1:N, i);
    Phi{i} = get_graphL(permute(Y, [mods, i]), 5);
    inv_Phi{i} = ((theta/beta_3)*Phi{i}+eye(size(Phi{i})))^-1;
end
D = convmtx([1,-1], size(Y,1));
D(:,end) = [];
D(end,1) = -1;
if beta_5~=0 && beta_4~=0
    invD = (beta_5*eye(sz(1))+beta_4*(D'*D))^-1;
else
    invD = zeros(sz(1));
end
Lam{2} = cell(1, N);
for i=1:N
    Lam{2}{i} = zeros(size(Y));
end
Lam{3} = Lam{2};
Lam{1} = zeros(size(Y));
Lam{4} = Lam{1};
Lam{5} = Lam{1};

times = [];
iter = 1;
nuc_norm = num2cell([0,0,0,0]);
obj_val = compute_obj(Y,L,S,Lx,G,Nt,W,Z,Lam,D,Phi,param,nuc_norm);
while true
    %% L Update
    tstart = tic;
    temp = zeros(size(Y));
    for i=1:4
        temp = temp + beta_2*(Lx{i}-Lam{2}{i})+beta_3*(G{i}+Lam{3}{i});
    end
    T1 = Y-S+Lam{1}; 
    L(mask_Y) = (beta_1*T1(mask_Y)+temp(mask_Y))/(beta_1+4*(beta_2+beta_3));
    L(~mask_Y) = temp(~mask_Y)/(4*(beta_2+beta_3));
    times(iter,1) = toc(tstart);
    
    %% Lx Update
    tstart = tic;
    [Lx, nuc_norm] = soft_hosvd(L, Lam{2}, psi, 1/beta_2);
    times(iter,2) = toc(tstart);
    %% G Update
    tstart = tic;
    G = graph_reg_update2(L, Lam{3}, inv_Phi);
    times(iter,3) = toc(tstart); 
    %% S Update
    tstart = tic;
    temp1 = beta_1*(Y-L-Nt+Lam{1});
    temp1(~mask_Y) = 0;
    temp2 = beta_5*(W+Lam{5});
    Sold = S;
    S = soft_threshold((temp1+temp2), lambda)./(beta_1+beta_5);
    S(~mask_Y) = soft_threshold(temp2(~mask_Y), lambda)./beta_5;
    times(iter,4) = toc(tstart);
    %% N update
    tstart = tic;
    Nt = (beta_1/(beta_1+alpha)).*(Y+Lam{1}-L-S);
    Nt(~mask_Y) = 0;
    times(iter,5) = toc(tstart);
    %% W Update
    tstart = tic;
    W = invD*(beta_5*Runfold(S-Lam{5})+beta_4*D'*Runfold(Z+Lam{4}));
    W = reshape(W, sz);
    times(iter,6) = toc(tstart);
    
    %% Z Update
    tstart = tic;
    Z = soft_threshold(mergeTensors(D, W, 2, 1)-Lam{4}, gamma/(beta_4+eps));
    times(iter,7) = toc(tstart);
    %% Dual Updates
    tstart = tic;
    temp = 0;
%     temp2 = 0;
    for i=1:N
        Lam2_up = L-Lx{i};
        temp = temp + norm(Lam2_up(:))^2;
        Lam{2}{i} = Lam{2}{i}+Lam2_up;
        Lam3_up = G{i}-L;
%         temp2 = temp2 + norm(Lam3_up(:))^2;
        Lam{3}{i} = Lam{3}{i}+Lam3_up;
    end
    Lam{1} = Lam{1} + Y-L-S-Nt;
    Lam{1}(~mask_Y) = 0;
    temp = sqrt(temp)/(sqrt(N)*norm(Y(:)));
%     temp2 = sqrt(temp2)/sqrt(N)/norm(Y(:));
    Lam{4} = Lam{4} - mergeTensors(D, W, 2, 1) + Z;
    Lam{5} = Lam{5} - S + W;
    times(iter,8) = toc(tstart);
    
    %% Error and objective calculations
    obj_val(iter+1) = compute_obj(Y,L,S,Lx,G,Nt,W,Z,Lam,D,Phi,param, nuc_norm);
    err = max([norm(S(:)-Sold(:))/norm(Sold(:)), temp]);
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

function [val, term] = compute_obj(Y,L,S,Lx,G,Nt,W,Z,Lam,D,Phi,param, nuc_norm)

alpha = param.alpha;
theta = param.theta;
lambda = param.lambda;
gamma = param.gamma;
psi = param.psi;
beta_1 = param.beta_1;
beta_2 = param.beta_2;
beta_3 = param.beta_3;
beta_4 = param.beta_4;
beta_5 = param.beta_5;
N = length(Lx);
term = zeros(1,10);

for i=1:N
    term(1) = term(1) + psi(i)*nuc_norm{i};
    term(2) = term(2) + theta*comp_gr_reg(G{i}, Phi{i}, i);
    term(7) = term(7) + beta_2/2*sum((Lx{i}-L-Lam{2}{i}).^2,'all');
    term(8) = term(8) + beta_3/2*sum((L-G{i}-Lam{3}{i}).^2,'all');
end
term(6) = beta_1/2*sum((Y-S-L-Nt+Lam{1}).^2,'all');
term(9) = (beta_4/2)*sum((mergeTensors(D, W, 2, 1)-Lam{4}-Z).^2,'all');
term(10) = (beta_5/2)*sum((S-W-Lam{5}).^2,'all');
term(3) = lambda*sum(abs(S),'all');
term(5) = (alpha/2)*norm(Nt(:))^2;
term(4) = gamma*sum(abs(Z),'all');
val = sum(term);
end