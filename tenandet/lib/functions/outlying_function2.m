function [O,Or] = outlying_function2(Z, Sn,  maxit, err_tol,v)
% Sn: the cell of tensor samples
% z: Point of evaluation

n = length(Sn);
sz= size(Z);
N= length(sz);
u= cell(N,1);
z= cell(N,1);
sn= cell(n,1);
A= cell(N,1); % A = (x-E[X])(x-E[X])^T
B= cell(N,1); % B = E[(x-E[X])(x-E[X])^T]
O_R = zeros(maxit*N,1);


for i = 1:N
    u{i}= randn(sz(i),1);
    u{i}= u{i}/norm(u{i},2);
    A{i}= zeros(sz(i));
    B{i}= zeros(sz(i));
    mu{i}= zeros(sz(i),1);
    z{i}= zeros(sz(i),1);
end

it = 0;
while it<maxit
    for mode = 1:N  % for every mode
        
        
        z{mode} = Z;    % Transformed z
        tmp_sz = sz;
        for ii = 1:N
        
               if ii~=mode
                   tmp_sz(ii) = 1;
                   z{mode} = m2t( u{ii}.' * t2m(z{mode},ii) , tmp_sz, ii);
               end
        end
        z{mode}=reshape(z{mode},[sz(mode),1]);
        
        for i = 1:n % Transform other samples S_n to s_n (vectorize them)
            tmpX = Sn{i};
            tmp_sz = sz;
            for ii = 1:N % Inner product in other modes
               if ii~=mode
                   tmp_sz(ii) = 1;
                   tmpX = m2t( u{ii}.' * t2m(tmpX,ii) , tmp_sz, ii);
               end
            end
            sn{i}=reshape(tmpX,[sz(mode),1]);
        end

        mu{mode}= mean([sn{:}],2);
        B{mode}= cov([sn{:};].');
        A{mode}= (z{mode}-mu{mode})*(z{mode}-mu{mode}).';
        
        [V,D,W] = eig(A{mode},B{mode},'qz');
        [lda,idx] = max(diag(D));
        u{mode}=V(:,idx)/norm(V(:,idx),2);
        O_R(it*N+mode)= sqrt(lda);
    end
    if (v==1)&&(it>1) && ( abs( O_R(it*N+mode) -O_R(it*N+mode-N))<err_tol )
        disp("Converged")
        break
    end
    it = it+1;
end
O = O_R(it*N);
Or = O_R(1:it*N);
end