function [S, precision, recall, fpr, time_o]= ocsvm_subscript(Y, X, param)
% Computes anomaly scores for each  third mode fiber using One Class SVM.
out_fr = 0.1;
tic;
S = one_class_svm(Y, out_fr);
time_o = toc;

% Top-K Analysis
[~, precision, recall, fpr] = analyze_top_K(S, X, param.ind_m, true);
end