function X = lreLU(X, varargin)
% lreLU Leaky Restricted Linear Unit.
% Multiplies all negative elements by constant `leak`.
if isempty(varargin)
    leak = 0.1;
else
    leak = varargin{1};
end
X(X<0) = X(X<0).*leak;
end