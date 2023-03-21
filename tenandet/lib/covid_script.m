covid_data = load('covid_19data.mat', 'tensor');

Y = reshape(covid_data.tensor(:,1,:), [],83);

param.lambda = 1/sqrt(max(size(Y)));
param.gamma = 1/5/param.lambda;
param.alpha = 1/5/param.lambda;
param.beta_1 = 1/(5*std(Y(:)));
param.beta_2 = param.beta_1;
param.beta_3 = param.beta_1;
param.max_iter = 100;
param.err_tol = 0.0001;

[L,S] = low_temp_sp_dec(Y, param);
plot_covid(Y, S, L, 67)