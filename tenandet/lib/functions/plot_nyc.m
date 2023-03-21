function plot_nyc(Y, S1, S2, S3, S4, S5, S6, S7, day_id, lst_figs)
%UNTITLED2 Summary of this function goes here
%  Detailed explanation goes here
Y = permute(Y,[1,3,4,2]);
S1 = permute(S1,[1,3,4,2]);
S2 = permute(S2,[1,3,4,2]);
S3 = permute(S3,[1,3,4,2]);
S4 = permute(S4,[1,3,4,2]);
S5 = permute(S5,[1,3,4,2]);
S6 = permute(S6,[1,3,4,2]);
S7 = permute(S7,[1,3,4,2]);
if nargin==7
    num_figs = min(15,size(Y,3));
    lst_figs = 1:num_figs;
else
    num_figs = length(lst_figs);
end
for i=1:num_figs
    figure,
    subplot(2,4,1)
    plot(Y(:,:,lst_figs(i),day_id))
    title('HORPCA')
    subplot(2,4,2)
    plot(S1(:,:,lst_figs(i),day_id))
    title('GLOSS')
    subplot(2,4,3)
    plot((S2(:,:,lst_figs(i),day_id)))
    title('WHORPCA')
    subplot(2,4,4)
    plot((S3(:,:,lst_figs(i),day_id)))
    title('LOSS')
    subplot(2,4,5)
    plot((S5(:,:,lst_figs(i),day_id)))
    title('Data')
    subplot(2,4,6)
    plot((S4(:,:,lst_figs(i),day_id)))
    title('GLOSS-L')
    subplot(2,4,7)
    plot((S6(:,:,lst_figs(i),day_id)))
    title('WHORPCA-L')
    subplot(2,4,8)
    plot((S7(:,:,lst_figs(i),day_id)))
    title('LOSS-L')
end

end