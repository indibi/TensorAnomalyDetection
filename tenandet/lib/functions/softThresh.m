function X = softThresh(X, sigma)
%softThresh Soft thresholding function
%   X = softThresh(X, sigma)
X(abs(X)<sigma) = 0;
X(abs(X)>sigma) = X(abs(X)>sigma) - sign(X(abs(X)>sigma)).*sigma;
end