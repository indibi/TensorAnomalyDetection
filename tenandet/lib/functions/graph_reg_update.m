function La = graph_reg_update(L,Lam,inv_matrix)
%
%
N = length(L);
La = cell(1,N);
for i=1:N
    La{i} = Lunfold(L{i}-Lam{i}, N)*inv_matrix;
    La{i} = reshape(La{i}, size(L{i}));
end

end