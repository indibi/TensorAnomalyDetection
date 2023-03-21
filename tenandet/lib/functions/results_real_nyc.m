function results_real_nyc(S_gloss, S_lrt, S_whorpca, S_horpca, Y, a_L, OCS, ind_removed, param, w_mahal)
%% results_real_nyc(S_gloss, S_lrt, S_whorpca, S_horpca, Y, a_L, OCS, ind_removed, param, w_mahal)
% Function that plots the detection tables for all methods.
% !!Obsolete!!
list_k = [20,100,500,1000,2000,5000,7000,14000,21000,3*10^4:2*10^4:9*10^4];
det_gloss = zeros(length(list_k),20);
det_svm = det_gloss;
det_gloss_o = det_gloss;
det_gloss_l = det_gloss;
det_lrt = det_gloss;
det_whorpca = det_gloss;
det_horpca = det_gloss;
det_ee = det_gloss;
det_lof = det_gloss;
Y = mahal_dist(Y);
out_fr = 0.1;
S_gloss_a = one_class_svm(S_gloss, out_fr);
S_gloss_l = apply_lof(S_gloss, 10);
if w_mahal
    S_gloss = mahal_dist(S_gloss);
    S_lrt = mahal_dist(S_lrt);
    S_whorpca = mahal_dist(S_whorpca);
    S_horpca = mahal_dist(S_horpca);
end
for i=1:length(list_k)
    det_gloss_o(i,:) = (detect_real_events(S_gloss_a, ind_removed, list_k(i)));
    det_gloss_l(i,:) = (detect_real_events(S_gloss_l, ind_removed, list_k(i)));
    det_gloss(i,:) = (detect_real_events(S_gloss, ind_removed, list_k(i)));
    det_lrt(i,:) = (detect_real_events(S_lrt, ind_removed, list_k(i)));
    det_whorpca(i,:) = (detect_real_events(S_whorpca, ind_removed, list_k(i)));
    det_ee(i,:) = (detect_real_events(Y, ind_removed, list_k(i)));
    det_horpca(i,:) = (detect_real_events(S_horpca, ind_removed, list_k(i)));
    det_lof(i,:) = (detect_real_events(a_L, ind_removed, list_k(i)));
    det_svm(i,:) = (detect_real_events(OCS, ind_removed, list_k(i)));
end


figure,
hold
plot(100*list_k/numel(Y), sum(det_ee, 2),'Color','#0072BD','DisplayName','EE','LineWidth',7)
plot(100*list_k/numel(Y), sum(det_lof, 2),'Color','#D95319','DisplayName','LOF','LineWidth',7)
plot(100*list_k/numel(Y), sum(det_svm, 2),'Color','#EDB120','DisplayName','OCSVM','LineWidth',7)
plot(100*list_k/numel(Y), sum(det_horpca, 2),'Color','#7E2F8E','DisplayName','HORPCA-EE','LineWidth',7)
plot(100*list_k/numel(Y), sum(det_whorpca, 2),'Color','#77AC30','DisplayName','WHORPCA-EE','LineWidth',7)
plot(100*list_k/numel(Y), sum(det_lrt, 2),'--+','Color','#0072BD','DisplayName','LOSS-EE','LineWidth',7,'MarkerSize',20)
plot(100*list_k/numel(Y), sum(det_gloss, 2),'--x','Color','#0072BD','DisplayName','GLOSS-EE','LineWidth',7,'MarkerSize',20)
plot(100*list_k/numel(Y), sum(det_gloss_o, 2),'--x','Color','#EDB120','DisplayName','GLOSS-SVM','LineWidth',7,'MarkerSize',20)
plot(100*list_k/numel(Y), sum(det_gloss_l, 2),'--x','Color','#D95319','DisplayName','GLOSS-LOF','LineWidth',7,'MarkerSize',20)
legend('location','southeast')
title(['\lambda ',num2str(param.lambda), ', \gamma ',num2str(param.gamma)])
grid
ax = gcf;
ax.CurrentAxes.FontSize=26;
ax.CurrentAxes.FontWeight = 'bold';
end