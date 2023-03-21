function La = graph_reg_update_2(L, Lam, inv_tens)
%
%
N = length(L);
La = cell(1,N);
for i=1:N
    for j=1:size(inv_tens,3)
        La{i}(:,:,j) = Lunfold(L{i}(:,j,:,:)-Lam{i}(:,j,:,:), N)*inv_tens(:,:,j);
    end
    La{i} = permute(reshape(La{i}, size(L{i},1),size(L{i},3),size(L{i},4),size(L{i},2)),[1,4,2,3]);
end
end