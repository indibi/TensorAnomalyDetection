clear
nodisp = true;
sizes = [24,7,52,25];
Y = [];
%%Uncomment for synthesizing using NYC data
load nyc_tensors.mat
% load nyc_bikedata.mat
load regions.mat
arrs = squeeze(sum(reshape(arrs,6,24,365,[]),1));
Y = double(reshape(arrs(:,1:364,regions), 24, 7, 52,[]));
% Y = double(reshape(arrs(:,1:364,1:81), 24, 7, 52,[]));
sizes = size(Y);
%%_________________________________________________________________________
%%Uncomment for synthesizing using traffic data
% [~, Y, ~] = get_traffic_data;
anom_list = [700];
len_list = [7];
amp_anom = [2.5];
% load(['preliminary_results\gloss_amp', num2str(amp_anom),'.mat'])
n_missing = round([0.2:.2:.6].*(prod(sizes)/24));%,0.05,0.1,0.2:.2:0.8
rank_list = [7];
psi_list = 10.^(-1);
theta_list = 10.^[-2];
lam_list_1 = (10.^[-2])/(max(size(Y)));
lam_list_2 = (10.^[0])/(numel(Y));
lam_list_3 = (10.^[0])/max(size(Y));
gam_list_1 = (10.^[-2])/(max(size(Y)));
gam_list_2 = (10.^[-2])/(numel(Y));
gam_list_3 = (10.^[5])/max(size(Y));
n_exp = 10;
len_lam = length(lam_list_1);
len_gam = length(theta_list);
len_rank = length(rank_list);
len_n = length(n_missing);
len_a = length(amp_anom);
% precision = cell(1, n_exp);
% recall = precision;
% fpr = recall;
% for i=1:n_exp
%     precision{i} = zeros(20, len_gam, len_lam, len_n, len_a, 10);
%     recall{i} = zeros(20, len_gam, len_lam, len_n, len_a, 10);
%     fpr{i} = zeros(20, len_gam, len_lam, len_n, len_a, 10);
% end
% time_ee = zeros(n_exp,len_a,len_n);
% time_lof = zeros(n_exp,len_a,len_n);
time_ocsvm = zeros(n_exp,len_a,len_n);
% time_horpca = cell(1,len_n);
% time_whorpca = cell(1,len_n);
% time_gloss = cell(1,len_n);
% time_logss = cell(1,len_n);

% geo_mean_std = geomean([1/(sum(std(t2m(Y,4),[],2).^2)),...
%     1/(sum(std(t2m(Y,3),[],2).^2)),...
%     1/(sum(std(t2m(Y,2),[],2).^2)),...
%     1/(sum(std(t2m(Y,1),[],2).^2))]);
for i=1:4
    std_m(i) = sum(std(t2m(Y,i)));
end
parfor i=1:n_exp
    for i_a=1:len_a
        for i_N = 1:len_n
            %% Generate Data
            num_anom = anom_list(1);
            len_anom = len_list(1);
            [X, Y_gen, Yn, ind_removed, mat_anom] = gendata(sizes, num_anom, len_anom, amp_anom(i_a), Y, n_missing(i_N), false, i);
            %% EE
            tic;
            [mah_Yn, L_ee] = mahal_dist(Yn);
            time_ee(i, i_a, i_N) = toc;
            [~, precision{i}(i_a,:,1,1,i_N,1),...
                recall{i}(i_a,:,1,1,i_N,1), ...
                fpr{i}(i_a,:,1,1,i_N,1)] =...
                analyze_top_K(mah_Yn, X, ind_removed);
            %% LOF
            param =[];
            param.ind_m = ind_removed;
            [~, precision{i}(i_a,:,1,1,i_N,2),...
                recall{i}(i_a,:,1,1,i_N,2), ...
                fpr{i}(i_a,:,1,1,i_N,2), ...
                time_lof(i, i_a, i_N)] = ...
                lof_subscript(Yn, X, param);
            %% OCSVM
            [~, precision{i}(i_a,:,1,1,i_N,3),...
                recall{i}(i_a,:,1,1,i_N,3),...
                fpr{i}(i_a,:,1,1,i_N,3),...
                time_ocsvm(i, i_a, i_N)] =...
                ocsvm_subscript(Yn, X, param);
            %%
            for i_L=1:len_lam
                %% HORPCA
                param = [];
                param.psi = ones(1, 4);
                param = set_params(Yn, lam_list_2(i_L));
                param.lambda = 1/sqrt(max(size(Yn)));
                [~, ~, precision{i}(i_a,:,1,i_L,i_N,4),...
                    recall{i}(i_a,:,1,i_L,i_N,4),...
                    fpr{i}(i_a,:,1,i_L,i_N,4),...
                    time_horpca{i}(i_a,i_N,i_L)] = ...
                    horpca_subscript(Yn, Y_gen, X, param);
                %% WHORPCA
                param.psi = min(std_m)*std_m.^-1;
                param = set_params(Yn, lam_list_2(i_L));
                [~, ~, precision{i}(i_a,:,1,i_L,i_N,5),...
                    recall{i}(i_a,:,1,i_L,i_N,5), ...
                    fpr{i}(i_a,:,1,i_L,i_N,5), ...
                    time_whorpca{i}(i_a,i_N,i_L)] = ...
                    horpca_subscript(Yn, Y_gen, X, param);
                %% CP Based methods
                param = set_params(Yn, lam_list_3(i_L));
                for i_R = 1:len_rank
                    param.init_rank = rank_list(i_R);
                    [~, ~, precision{i}(i_a,:,i_R,i_L,i_N,6),...
                        recall{i}(i_a,:,i_R,i_L,i_N,6), ...
                        fpr{i}(i_a,:,i_R,i_L,i_N,6), ...
                        time_cp{i}(i_a,i_N,i_L,i_R)] = ...
                        cp_subscript(Yn, Y_gen, X, param);
                    % BATF
                    [~, ~, precision{i}(i_a,:,i_R,i_L,i_N,11),...
                        recall{i}(i_a,:,i_R,i_L,i_N,11), ...
                        fpr{i}(i_a,:,i_R,i_L,i_N,11), ...
                        time_batf{i}(i_a,i_N,i_L,i_R)] = ...
                        batf_subscript(Yn, Y_gen, X, param);
                end

                for i_G =1:len_gam
                    %% LOSS
                    param.lambda = lam_list_1(i_L);
                    param.psi = min(std_m)*std_m.^-1;
                    param.gamma = gam_list_1(1);
                    [~,~,precision{i}(i_a,:,i_G,i_L,i_N,7),...
                        recall{i}(i_a,:,i_G,i_L,i_N,7),...
                        fpr{i}(i_a,:,i_G,i_L,i_N,7),...
                        times_loss{i}(i_a,i_N,i_L,i_G)] = ...
                        loss_subscript(Yn, Y_gen, X, param);
                    %% GLOSS
                    param.theta = geomean([1/(sum(std(t2m(Y,4),[],2).^2)),...
                            1/(sum(std(t2m(Y,3),[],2).^2)),...
                            1/(sum(std(t2m(Y,2),[],2).^2)),...
                            1/(sum(std(t2m(Y,1),[],2).^2))]);
                    param.lambda = lam_list_2(1);
                    param.gamma = gam_list_2(1);
                    param.theta = param.theta*theta_list(i_G);
                    [L_gloss, S_gloss, precision{i}(i_a,:,i_G,i_L,i_N,8:10),...
                        recall{i}(i_a,:,i_G,i_L,i_N,8:10),...
                        fpr{i}(i_a,:,i_G,i_L,i_N,8:10),...
                        times_gloss{i}(i_a,i_N,i_L,i_G)] = ...
                        gloss_subscript(Yn, Y_gen, X, param);
%                     %% LOGSS
%                     param.lambda = lam_list_3(i_L);
%                     param.gamma = gam_list_3(i_G);
%                     param.theta = theta_list(1);
%                     [L_log, S_log, precision{i}(i_a,:,i_G,i_L,i_N,11), ...
%                         recall{i}(i_a,:,i_G,i_L,i_N,11), ...
%                         fpr{i}(i_a,:,i_G,i_L,i_N,11),...
%                         times_logss{i}(i_a,i_N,i_L,i_G)] = ...
%                         logss_subscript(Yn, Y_gen, X, param);
                end
            end
        end
    end
end

[auc, times] = plot_roc(fpr, recall,...
    time_ee,time_lof,time_ocsvm,time_horpca,time_whorpca,time_cp,...
    times_loss,times_gloss, time_batf);
% [auc, times] = plot_lam_gam(fpr, recall);
save(['preliminary_results\miss.mat'], 'recall', 'fpr',...
    'gam_list_1', 'gam_list_2', 'gam_list_3', 'lam_list_1', 'lam_list_2',...
    'lam_list_3', 'amp_anom', 'n_missing', 'auc', 'times')