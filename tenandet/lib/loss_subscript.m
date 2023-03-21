function [L,S,precision, recall, fpr, time, rmse, mape] = loss_subscript(Y, Y_gen, X, param)

% param.lambda = 1/4/sqrt(max(size(Yn)));
% param.lambda = 1/2/sqrt(max(size(Y)));
% param.gamma = 1/15/sqrt(max(size(Y)));
% param.err_tol = 0.001;

t = tic;
[L, S, ~] = loss(Y, param);
rmse = norm(L(:)-Y_gen(:))/sqrt(numel(Y_gen));
mape = sum(abs(L-Y_gen),'all')/numel(Y_gen);
time = toc(t);

%% Top-K Analysis
[~, precision, recall, fpr] = analyze_top_K(mahal_dist(S), X, param.ind_m);
%% Visualized Decomposition
% if ind_outer == length(anom_list)
% %     plot_sensor_new(permute(X,[1,3,4,2]), permute(Yn,[1,3,4,2]), permute(mahal_S,[1,3,4,2]), permute(S_lrt,[1,3,4,2]), 3, 10)
% end
end