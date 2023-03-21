function param = set_params(Y, varargin)
% param = set_params(Y, lambda, gamma, theta, psi, beta, ind_missing, err_tol, max_iter, disp)
% Function that sets parameters and returns them in a struct.
% Inputs:
%   Y: Data tensor to be learned, some parameters are related to the size
%   and standard deviations.
param.alpha = 0;
param.lambda = 1/sqrt(max(size(Y)));
param.gamma = 1/numel(Y);
param.mu = 10^-3;
N = ndims(Y);
for i=1:N
    std_modes(i) = 1/sum(std(t2m(Y,i),[],2).^2);
end
geo_mean_std = geomean(std_modes);
param.theta = geo_mean_std;
for i=1:N
    std_m(i) = sum(std(t2m(Y,i)));
end
param.psi = max(std_m)*std_m.^-1;
param.beta_1 = 1/(5*std(Y(:)));
param.beta_2 = param.beta_1;
param.beta_3 = param.beta_1;
param.beta_4 = param.beta_1;
param.beta_5 = param.beta_1;

param.ind_m = [];
param.opt_tol = 0.01;
param.err_tol = 0.01;
param.max_iter = 100;
param.disp = 0;

if nargin >= 2
    param.lambda = varargin{1};
end

if nargin >= 3
    param.gamma = varargin{2};
end

if nargin >= 4
    param.theta = varargin{3};
end

if nargin >= 5
    param.psi = varargin{4};
end

if nargin >= 6
    param.beta_1 = varargin{5};
    param.beta_2 = varargin{5};
    param.beta_3 = varargin{5};
    param.beta_4 = varargin{5};
    param.beta_5 = varargin{5};
end

if nargin >= 7
    param.ind_missing = varargin{6};
end

if nargin >= 8
    param.err_tol = varargin{7};
end

if nargin >= 9
    param.max_iter = varargin{8};
end

if nargin >= 10
    param.disp = varargin{9};
end

end