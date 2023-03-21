function [auc, f, r, times] = plot_roc(fpr,  recall, varargin)
% auc = plot_roc(fpr, recall)
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
    times_batf = varargin{9};
end

n_exp = length(recall);
f = 0;
r = 0;
for i=1:n_exp
    f = f+permute(fpr{i},[2,3,4,5,1,6])./n_exp;
    r = r+permute(recall{i},[2,3,4,5,1,6])./n_exp;
end
len_gam = size(f,2);
len_lam = size(f,3);
len_n = size(f,4);
len_a = size(f,5);
auc = zeros(len_n, len_gam, len_lam, len_a, 11, 10);

for l = 1:len_a
    for i = 1:len_n
        for j = 1:len_gam
            for k = 1:len_lam
                figure,
                plot(f(:,1,1,i,l,1), r(:,1,1,i,l,1),'Color','#0072BD','DisplayName','EE','LineWidth',7);
                hold on;
                plot(f(:,1,1,i,l,3), r(:,1,1,i,l,3),'Color','#EDB120','DisplayName','OCSVM','LineWidth',7)
                plot(f(:,1,1,i,l,2), r(:,1,1,i,l,2),'Color','#D95319','DisplayName','LOF','LineWidth',7);
                plot(f(:,1,k,i,l,4), r(:,1,k,i,l,4),'Color','#7E2F8E','DisplayName','HORPCA','LineWidth',7);
                plot(f(:,1,k,i,l,5), r(:,1,k,i,l,5),'--','Color','#77AC30','DisplayName','WHORPCA','LineWidth',7);
                plot(f(:,j,k,i,l,6), r(:,j,k,i,l,6),'--o','Color','#A2142F','DisplayName','CP','LineWidth',7,'MarkerSize',20);
                plot(f(:,j,k,i,l,11), r(:,j,k,i,l,11),'--x','Color','#A2142F','DisplayName','BATF','LineWidth',7,'MarkerSize',20)
                plot(f(:,j,k,i,l,7), r(:,j,k,i,l,7),'--+','Color','#4DBEEE','DisplayName','LOSS','LineWidth',7,'MarkerSize',20);
                plot(f(:,j,k,i,l,8), r(:,j,k,i,l,8),'--x','Color','#0072BD','DisplayName','GLOSS-EE','LineWidth',7,'MarkerSize',20)
                plot(f(:,j,k,i,l,9), r(:,j,k,i,l,9),'--x','Color','#EDB120','DisplayName','GLOSS-SVM','LineWidth',7,'MarkerSize',20)
                plot(f(:,j,k,i,l,10), r(:,j,k,i,l,10),'--x','Color','#D95319','DisplayName','GLOSS-LOF','LineWidth',7,'MarkerSize',20)
% zz                plot(f(:,j,k,i,l,10), r(:,j,k,i,l,12),'--x','Color','#A2142F','DisplayName','LOGSS','LineWidth',7,'MarkerSize',20)
                legend('location','southeast'), grid
%                 title([num2str(round(100*24*n_missing(i)/prod(sizes))), '\lambda = ', num2str(lam_list_2(k)), '\gamma = ', num2str(gam_list_2(j)), ' 0.5 std'])
                ax = gcf;
                ax.CurrentAxes.FontSize = 26;
                ax.CurrentAxes.FontWeight = 'bold';
                for i_n = 1:n_exp
                    auc(i,j,k,l,1,i_n) = trapz(fpr{i_n}(l,:,1,1,i,1), recall{i_n}(l,:,1,1,i,1));
                    auc(i,j,k,l,2,i_n) = trapz(fpr{i_n}(l,:,1,1,i,3), recall{i_n}(l,:,1,1,i,3));
                    auc(i,j,k,l,3,i_n) = trapz(fpr{i_n}(l,:,1,1,i,2), recall{i_n}(l,:,1,1,i,2));
                    auc(i,j,k,l,4,i_n) = trapz(fpr{i_n}(l,:,1,k,i,4), recall{i_n}(l,:,1,k,i,4));
                    auc(i,j,k,l,5,i_n) = trapz(fpr{i_n}(l,:,1,k,i,5), recall{i_n}(l,:,1,k,i,5));
                    auc(i,j,k,l,6,i_n) = trapz(fpr{i_n}(l,:,j,k,i,6), recall{i_n}(l,:,j,k,i,6));
                    auc(i,j,k,l,7,i_n) = trapz(fpr{i_n}(l,:,j,k,i,7), recall{i_n}(l,:,j,k,i,7));
                    auc(i,j,k,l,8,i_n) = trapz(fpr{i_n}(l,:,j,k,i,8), recall{i_n}(l,:,j,k,i,8));
                    auc(i,j,k,l,9,i_n) = trapz(fpr{i_n}(l,:,j,k,i,9), recall{i_n}(l,:,j,k,i,9));
                    auc(i,j,k,l,10,i_n) = trapz(fpr{i_n}(l,:,j,k,i,10), recall{i_n}(l,:,j,k,i,10));
                    auc(i,j,k,l,11,i_n) = trapz(fpr{i_n}(l,:,j,k,i,11), recall{i_n}(l,:,j,k,i,11));
                end

                if ~isempty(varargin)
                    for i_n = 1:n_exp
%                         times(i,j,k,l,i_n,1) = time_ee(i_n,l,i);
%                         times(i,j,k,l,i_n,2) = time_lof(i_n,l,i);
%                         times(i,j,k,l,i_n,3) = time_ocsvm(i_n,l,i);
                        times(i,j,k,l,i_n,4) = time_horpca{i_n}(l, i, k);
                        times(i,j,k,l,i_n,5) = time_whorpca{i_n}(l, i, k);
                        times(i,j,k,l,i_n,6) = time_cp{i_n}(l, i, k);
                        times(i,j,k,l,i_n,7) = times_loss{i_n}(l, i, k);
                        times(i,j,k,l,i_n,8) = times_gloss{i_n}(l, i, k);
                        times(i,j,k,l,i_n,9) = times_batf{i_n}(l, i, k);
                    end
                end
            end
        end
    end
end

end