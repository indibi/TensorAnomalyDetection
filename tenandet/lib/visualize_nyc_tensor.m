
sz = size(Y);

weekdays = [1,2,6,7];
wd_str = {'Monday','Tuesday','Saturday','Sunday'};
l_w = length(weekdays);
zones = [1,25,54,75];
l_z = length(zones);
for i=1:l_w
%     for j=1:l_z
%         subplot(1, l_w+1, i+floor((i-1)*2/l_w))
%         imagesc(squeeze(Y(:,weekdays(i),:,zones(3)))')
%         title(wd_str{i})
%         xlabel('Hours')
%         ylabel('Weeks')
        imwrite( ind2rgb(imresize(im2uint8(squeeze(Y(:,:,weekdays(i),zones(3)))/max(max(Y(:,:,weekdays(i),zones(3))))),[480,640],'method','nearest'), parula(256)), ['Zone_54_day', num2str(i),'.png'])
        imwrite( ind2rgb(imresize(im2uint8(squeeze(L_gloss(:,:,weekdays(i),zones(3)))/max(max(L_gloss(:,:,weekdays(i),zones(3))))),[480,640],'method','nearest'), parula(256)), ['Zone_54_day', num2str(i),'_lr.png'])
        imwrite( ind2rgb(imresize(im2uint8(squeeze(S_gloss(:,:,weekdays(i),zones(3)))/max(max(S_gloss(:,:,weekdays(i),zones(3))))),[480,640],'method','nearest'), parula(256)), ['Zone_54_day', num2str(i),'_sp.png'])
        % plot(squeeze(Y(:,weekdays(i),:,zones(j))))
        % title(['All ',wd_str{i},'s'])
        % xlabel('Hours')
        % ylabel('Number of Arrivals')
%     end
end