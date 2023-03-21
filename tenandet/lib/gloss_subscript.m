function [L, S, precision, recall, fpr, time_g, times, rmse, mape] = gloss_subscript(Y, Y_gen, X, param)
%% [L, S, precision, recall, fpr, time_g, times, rmse, mape] = gloss_subscript(Y, Y_gen, X, param)
% Script that applies GLOSS, and the following anomaly detection using: EE,
% LOF, OCSVM.
%
%  Parameters:
%   Y: Noisy data with missing entries.
%   Y_gen: Clean, fully observed data.
%   X: Anomaly mask
%   param: struct with field ind_m, corresponding to the mask for missing
%   data.
%
%  Returns:
%   L: Low rank tensor
%   S: Sparse tensor
%   precision: Precision scores for 20 different percentage levels, for
%   each anomaly detection algorithm
%   recall: Recall
%   fpr: False Positive Ratio
%   time_g: Total computation time of GLOSS
%   times: Array that analyzes time complexity of each update of each
%   variable, e.g. kth iteration, S update.
%   rmse: Root Mean Square Error between L and original, clean data
%   mape: Maximum A Posteriori Error between L and original, clean data

tic;
[L,S,~,times] = gloss(Y, param);
time_g = toc;
rmse = norm(L(:)-Y_gen(:))/sqrt(numel(Y_gen));
mape = sum(abs(L-Y_gen),'all')/numel(Y_gen);


out_fr = 0.1;
S_svm = one_class_svm(S, out_fr);
% 
S_lof = apply_lof(S, 10);
% Top-K Analysis
[~, precision, recall, fpr] = analyze_top_K(mahal_dist(S), X, param.ind_m);
[~, precision(:,2), recall(:,2), fpr(:,2)] = analyze_top_K(S_svm, X, param.ind_m, true);
[~, precision(:,3), recall(:,3), fpr(:,3)] = analyze_top_K(S_lof, X, param.ind_m);
end