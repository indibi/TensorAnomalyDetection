function [L, S, precision, recall, fpr, time_l, times, rmse, mape] =  logss_subscript(Y, Y_gen, X, param)

tic;
[L, S, times] = logss(Y, param);
time_l = toc;
rmse = norm(L(:)-Y_gen(:))/sqrt(numel(Y_gen));
mape = sum(abs(L-Y_gen),'all')/numel(Y_gen);

% out_fr = 0.1;
% S_svm = one_class_svm(S, out_fr);
% 
% S_lof = apply_lof(S, 10);
% Top-K Analysis
[~, precision, recall, fpr] = analyze_top_K(mahal_dist(S), X, param.ind_m);
% [~, precision(:,2), recall(:,2), fpr(:,2)] = analyze_top_K(S_svm, X, param.ind_m, true);
% [~, precision(:,3), recall(:,3), fpr(:,3)] = analyze_top_K(S_lof, X, param.ind_m);
end