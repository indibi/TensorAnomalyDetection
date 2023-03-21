function La = graph_reg_update2(L,Lam,inv)
%
%
N = length(Lam);
La = cell(1,N);
for i=1:N
    La{i} = inv{i}*t2m(L-Lam{i}, i);
    La{i} = m2t(La{i}, size(L), i);
end

end