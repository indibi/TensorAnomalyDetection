function [L, U] = cp_fac_update_als(Y, U, ind_m, mu)

N = ndims(Y);
R = size(U{1},2);
sz = size(Y);
if isempty(ind_m)
    i = [];
else
    [i(:,1),i(:,2),i(:,3),i(:,4)] = ind2sub(sz, ind_m);
end
Ui = cell(1,N);
Uinv = cell(1,N);
for n=1:N
    Ui{n} = khatrirao(U([1:n-1,n+1:N]),'r');
    Uinv{n} = Ui{n}*(Ui{n}'*Ui{n}+mu*eye(R))^-1;
end
for n=1:4
    temp = t2m(Y,n);
    for i_n = 1:sz(n)
        if isempty(ind_m) || isempty(i_n==i(:,n))
            U{n}(i_n,:) = temp(i_n,:)*Uinv{n};
        else
            temp_ind = sub2ind(sz([1:n-1,n+1:N]), i(:,[1:n-1,n+1:N]));
            temp_ind = setdiff(prod(sz)/sz(n),temp_ind);
            temp_Ui = Ui{n}(temp_ind,:);
            
            U{n}(i_n,:) = (temp(i_n, temp_ind)*temp_Ui)/(temp_Ui'*temp_Ui);
        end
    end
end
L = khatrirao(U(1:N-1),'r');
L = reshape(L*U{N}',sz);
end