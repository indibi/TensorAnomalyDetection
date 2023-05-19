function U = unifFun(d,varargin)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    r_0 = 1;
    M=1000;
elseif nargin < 3
    r_0 = varargin{1};
    M = 1000;
elseif nargin == 3
    r_0 = varargin{1};
    M =  varargin{2};
end

U = zeros(d,M);
R = rand(M,1)*r_0;
for i =1:M
    u = randn(d,1);
    U(:,i) = R(i)*u/norm(u);
end
end

