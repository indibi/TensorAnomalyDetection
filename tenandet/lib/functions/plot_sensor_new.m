function plot_sensor_new(X, Y, S, S_lbl, checkSens, num_figs)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if nargin==5
    num_figs=min(15,size(S,3));
end
for i=1:num_figs
    figure,
    subplot(4,1,1)
    plot(X(:,:,i,checkSens))
    subplot(4,1,2)
    plot(Y(:,:,i,checkSens))
    subplot(4,1,3)
    plot((S(:,:,i,checkSens)))
    subplot(4,1,4)
    plot((S_lbl(:,:,i,checkSens)))
end

end