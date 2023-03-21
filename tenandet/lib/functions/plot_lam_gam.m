function [auc, times] = plot_lam_gam(fpr,  recall, varargin)
% auc = plot_lam_gam(fpr, recall)
% Plots ROC curves of various methods of many experiments, including
% experiments on parameters like missing rate, anomaly amplitude, lambda
% and gamma.
% Inputs: 
%  fpr: False positive (FP) rates. FP/(FN+TN);
%  recall: Recalls. TP/(TP+FN)
if ~isempty(varargin)
    time_ee = varargin{1};
    time_lof = varargin{2};
    time_ocsvm = varargin{3};
    time_horpca = varargin{4};
    time_whorpca = varargin{5};
    time_cp = varargin{6};
    times_loss = varargin{7};
    times_gloss = varargin{8};
    times_logss = varargin{9};
end

n_exp = length(recall);
len_gam = size(fpr{1},3);
len_n = size(fpr{1},5);
len_a = size(fpr{1},1);
len_lam = size(fpr{1},4);
auc = zeros(len_n, len_gam, len_lam, len_a, n_exp, 10);
times = zeros(len_n, len_gam, len_lam, len_a, n_exp, 8);
for i = 1:n_exp
    for i_L = 1:len_lam
        for i_N = 1:len_n
            for i_G = 1:len_gam
                for i_a = 1:len_a
%                     figure,
% %                     plot(fpr{i}(i_a,:,1,1,i_N,1), recall{i}(i_a,:,1,1,i_N,1),'Color','#0072BD','DisplayName','EE','LineWidth',7);
% %                     hold on;
% %                     plot(fpr{i}(i_a,:,1,1,i_N,3), recall{i}(i_a,:,1,1,i_N,3),'Color','#EDB120','DisplayName','OCSVM','LineWidth',7)
% %                     plot(fpr{i}(i_a,:,1,1,i_N,2), recall{i}(i_a,:,1,1,i_N,2),'Color','#D95319','DisplayName','LOF','LineWidth',7);
% %                     plot(fpr{i}(i_a,:,1,i_L,i_N,4), recall{i}(i_a,:,1,i_L,i_N,4),'Color','#7E2F8E','DisplayName','HORPCA','LineWidth',7);
% %                     plot(fpr{i}(i_a,:,1,i_L,i_N,5), recall{i}(i_a,:,1,i_L,i_N,5),'--','Color','#77AC30','DisplayName','WHORPCA','LineWidth',7);
% %                     plot(fpr{i}(i_a,:,i_G,i_L,i_N,6), recall{i}(i_a,:,i_G,i_L,i_N,6),'--+','Color','#4DBEEE','DisplayName','LOSS','LineWidth',7,'MarkerSize',20);
%                     plot(fpr{i}(i_a,:,i_G,i_L,i_N,7), recall{i}(i_a,:,i_G,i_L,i_N,7),'--x','Color','#0072BD','DisplayName','GLOSS-EE','LineWidth',7,'MarkerSize',20)
% %                     plot(fpr{i}(i_a,:,i_G,i_L,i_N,8), recall{i}(i_a,:,i_G,i_L,i_N,8),'--x','Color','#EDB120','DisplayName','GLOSS-SVM','LineWidth',7,'MarkerSize',20)
% %                     plot(fpr{i}(i_a,:,i_G,i_L,i_N,9), recall{i}(i_a,:,i_G,i_L,i_N,9),'--x','Color','#D95319','DisplayName','GLOSS-LOF','LineWidth',7,'MarkerSize',20)
% %                     plot(fpr{i}(i_a,:,i_G,i_L,i_N,10), recall{i}(i_a,:,i_G,i_L,i_N,10),'--x','Color','#A2142F','DisplayName','LOGSS','LineWidth',7,'MarkerSize',20)
%                     legend('location','southeast'), grid
%     %                 title(['\lambda : ', num2str(i_L), '\gamma : ', num2str(i_G), ' 0.5 std'])
%                     ax = gcf;
%                     ax.CurrentAxes.FontSize = 26;
%                     ax.CurrentAxes.FontWeight = 'bold';
%                     auc(i_N,i_G,i_L,i_a,i,1) = trapz(fpr{i}(i_a,:,1,1,i_N,1), recall{i}(i_a,:,1,1,i_N,1));
%                     auc(i_N,i_G,i_L,i_a,i,2) = trapz(fpr{i}(i_a,:,1,1,i_N,3), recall{i}(i_a,:,1,1,i_N,3));
%                     auc(i_N,i_G,i_L,i_a,i,3) = trapz(fpr{i}(i_a,:,1,1,i_N,2), recall{i}(i_a,:,1,1,i_N,2));
%                     auc(i_N,i_G,i_L,i_a,i,4) = trapz(fpr{i}(i_a,:,1,i_L,i_N,4), recall{i}(i_a,:,1,i_L,i_N,4));
%                     auc(i_N,i_G,i_L,i_a,i,5) = trapz(fpr{i}(i_a,:,1,i_L,i_N,5), recall{i}(i_a,:,1,i_L,i_N,5));
                    auc(i_N,i_G,i_L,i_a,i,6) = trapz(fpr{i}(i_a,:,i_G,i_L,i_N,6), recall{i}(i_a,:,i_G,i_L,i_N,6));
%                     auc(i_N,i_G,i_L,i_a,i,7) = trapz(fpr{i}(i_a,:,i_G,i_L,i_N,7), recall{i}(i_a,:,i_G,i_L,i_N,7));
                    auc(i_N,i_G,i_L,i_a,i,8) = trapz(fpr{i}(i_a,:,i_G,i_L,i_N,8), recall{i}(i_a,:,i_G,i_L,i_N,8));
%                     auc(i_N,i_G,i_L,i_a,i,9) = trapz(fpr{i}(i_a,:,i_G,i_L,i_N,9), recall{i}(i_a,:,i_G,i_L,i_N,9));
%                     auc(i_N,i_G,i_L,i_a,i,10) = trapz(fpr{i}(i_a,:,i_G,i_L,i_N,10), recall{i}(i_a,:,i_G,i_L,i_N,10));
                    
                    if ~isempty(varargin)
%                         times(i_N,i_G,i_L,i_a,i, 1) = time_ee(i,i_a,i_N);
%                         times(i_N,i_G,i_L,i_a,i, 2) = time_lof(i,i_a,i_N);
%                         times(i_N,i_G,i_L,i_a,i, 3) = time_ocsvm(i,i_a,i_N);
%                         times(i_N,i_G,i_L,i_a,i, 4) = time_horpca{i}(i_a, i_N, i_L);
%                         times(i_N,i_G,i_L,i_a,i, 5) = time_whorpca{i}(i_a, i_N, i_L);
                        times(i_N,i_G,i_L,i_a,i, 6) = time_cp{i}(i_a, i_N, i_L);
%                         times(i_N,i_G,i_L,i_a,i, 7) = times_loss{i}(i_a, i_N, i_L, i_G);
                        times(i_N,i_G,i_L,i_a,i, 8) = times_gloss{i}(i_a, i_N, i_L, i_G);
%                         times(i_N,i_G,i_L,i_a,i, 9) = times_logss{i}(i_a, i_N, i_L, i_G);
                    end
                end
            end
        end
    end
end


end