function [L, S, precision, recall, fpr, time_h, rmse, mape] = horpca_subscript(Y, Y_gen, X, param)
% param.err_tol = 0.001;
tic;
[L, S] = horpca(Y, param);
rmse = norm(L(:)-Y_gen(:))/sqrt(numel(Y_gen));
mape = sum(abs(L-Y_gen),'all')/numel(Y_gen);
time_h = toc;
%% Top-K Analysis
[~, precision, recall, fpr] = analyze_top_K(mahal_dist(S), X, param.ind_m);
end
