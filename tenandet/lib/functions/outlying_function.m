function O = outlying_function(z, Sn, M, maxit)
% Sn: the cell of tensor samples
% M: Number of vectors generated
% z: Point of evaluation
n = length(Sn); r_0 = 1;
sz = size(z);
N = length(sz);
U = cell(N,1);
W = cell(N,1);
m = cell(N,1);
s = cell(N,1);
t = cell(N,1); t_max = zeros(N,1);
Z = cell(N,1);
V = cell(N,1);

for i = 1:N
    U{i} = ones(sz(i),1);
    W{i} = zeros(n,M);
    m{i} = zeros(1,M);
    s{i} = zeros(1,M);
    t{i} = zeros(1,M);
    % Z{i} = zeros(sz(i),1);
    V{i} = zeros(sz(i), n);
end

it = 0;
while it<maxit
    for mode = 1:N  % for every mode
        
        
        Z{mode} = z;    % Transformed z
        tmp_sz = sz;
        for ii = 1:N
        
               if ii~=mode
                   tmp_sz(ii) = 1;
                   Z{mode} = m2t( U{ii}.' * t2m(Z{mode},ii) , tmp_sz, ii);
               end
        end
        Z{mode}=reshape(Z{mode},[sz(mode),1]);
        
        for i = 1:n % Calculate v_i^(mode)
            tmpX = Sn{i};
            tmp_sz = sz;
            for ii = 1:N % Inner product in other modes
               if ii~=mode
                   tmp_sz(ii) = 1;
                   tmpX = m2t( U{ii}.' * t2m(tmpX,ii) , tmp_sz, ii);
               end
            end
            V{mode}(:,i) = tmpX;
        end
        
        Urand = unifFun(sz(mode),r_0, M);
        for j = 1:M     % Calculate w_ij^(mode)
            for i = 1:n
                W{mode}(i,j) =  Urand(:,j).'* V{mode}(:,i);
            end
            m{mode}(j) = mean(W{mode}(:,j));
            s{mode}(j) = sqrt( ( (W{mode}(i,j)-m{mode}(j))^2 ) /n );
            t{mode}(j) = abs(Urand(:,j).'*Z{mode}-m{mode}(j))/s{mode}(j);
        end 
        [t_max(mode),max_idx] = max(t{mode});
        U{mode} = Urand(:,max_idx);
    end
    it = it+1;
end
O = max(t_max);
end