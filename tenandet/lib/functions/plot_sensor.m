function plot_sensor(X, Y, D, U, checkSens, modes)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
M = ndims(Y);
% m = length(modes);
% Yp = mergeTensors(Y, U(1:m), modes);
% Yp = ipermute(squeeze(mergeFactors({Yp, U{m+1:end}})),  [setdiff(1:M, modes), modes(end:-1:1)]);
imodes = setdiff(1:M, modes);
sz = size(Y);
Yp = reshape(U*t2m(Y, modes), [sz(modes), sz(imodes)]);
Yp = ipermute(Yp, [modes, imodes]);

for i=1:size(X,3)
figure,
subplot(3,2,1)
plot(X(:,:,i,checkSens))
subplot(3,2,2)
plot(Y(:,:,i,checkSens))
subplot(3,2,3)
plot(sign(Yp(:,:,i,checkSens)-D(:,:,i)))
subplot(3,2,4)
plot(Yp(:,:,i,checkSens)-D(:,:,i))
subplot(3,2,5)
plot(sign(Yp(:,:,i,checkSens)))
subplot(3,2,6)
plot(Yp(:,:,i,checkSens))
end

end

