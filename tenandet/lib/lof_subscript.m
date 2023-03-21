function [a_L, precision, recall, fpr, time_lof] = lof_subscript(Y, X, param)
%
%
tic;
a_L = apply_lof(Y, 10);
time_lof = toc;

% Top-K Analysis
[~, precision, recall,fpr] = analyze_top_K(a_L, X, param.ind_m);
end