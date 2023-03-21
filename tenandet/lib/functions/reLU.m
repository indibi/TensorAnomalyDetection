function X = reLU(X)
% reLU Restricted Linear Unit.
% Sets all negative elements to zero.
X(X<0) = 0;
end