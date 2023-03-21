clear
load nyc_bikedata.mat
load regions.mat
arrs = squeeze(sum(reshape(arrs,6,24,365,[]),1));
Y = double(reshape(arrs(:,1:364,:), 24, 7, 52,[]));
Y(:,1,53,:) = arrs(:,365,:);
Y(:,2:7,53,:) = mean(Y(:,2:7,1:52,:),3);
ind_removed = [];
for i=1:4
    std_m(i) = sum(std(t2m(Y,i)));
end
for k=1:4
    std_modes(k) = 1/sum(std(t2m(Y,k),[],2).^2);
end
geo_mean_std = geomean(std_modes);
list_k = [20,100,500,1000,2000,5000,7000,14000,21000,30000,50000,70000,90000];
lam_list = 10.^[-5:5];
gam_list = 10.^[-5:5];
rank_list = 1:11;
len_l = length(lam_list);
len_g = length(gam_list);
len_k = length(list_k);
det_gloss = zeros(20,len_l,len_g,len_k);
det_loss = det_gloss;
det_logss = det_gloss;
det_whorpca = det_loss;
det_horpca = det_loss;
det_ee = det_loss;
det_lof = det_loss;
det_ocsvm = det_loss;

tic;
ee = mahal_dist(Y);
time_ee = toc;
tic
a_L = apply_lof(Y, 10);
time_lof = toc;
tic;
out_fr = 0.1;
OCS = one_class_svm(Y, out_fr);
time_ocsvm = toc;
for k=1:len_k
    det_ee(:,1:len_l,1:len_g,k) = repmat(detect_real_events(ee, ind_removed, list_k(k))',1,len_l,len_g);
    det_lof(:,1:len_l,1:len_g,k) = repmat(detect_real_events(a_L, ind_removed, list_k(k))',1,len_l,len_g);
    det_ocsvm(:,1:len_l,1:len_g,k) = repmat(detect_real_events(OCS, ind_removed, list_k(k))',1,len_l,len_g);
end

for i=1:len_l
    param = [];
    param.ind_m = ind_removed;
    param.lambda = lam_list(i)/(max(size(Y)));
    param.alpha = 0;
    param.beta_1 = 1/(5*std(Y(:)));
    param.beta_2 = param.beta_1;
    param.beta_3 = param.beta_1;
    param.beta_4 = param.beta_1;
    param.max_iter = 100;
    param.err_tol = 0.01;
    param.psi = [1,1,1,1];
    [~,S_horpca, ~] = horpca(Y, param);
    param.psi = max(std_m)*std_m.^-1;%[0.1, 1,5,0.01];
    [L_whorpca,S_whorpca, ~] = horpca(Y, param);
    m_horpca = mahal_dist(S_horpca);
    m_whorpca = mahal_dist(S_whorpca);
    parfor j=1:len_g
        param = [];
        param.ind_m = ind_removed;
        param.theta = 1/(sum(std(t2m(Y,4),[],2).^2))^2;
        param.alpha = 0;
        param.disp = false;
        param.beta_1 = 1/(5*std(Y(:)));
        param.beta_2 = param.beta_1;
        param.beta_3 = param.beta_1;
        param.beta_4 = param.beta_1;
        param.beta_5 = param.beta_1;
        param.max_iter = 100;
        param.err_tol = 0.01;
        param.opt_tol = 0.01;
        param.psi = max(std_m)*std_m.^-1;
        param.mu = 10^-3;
        param.lambda = lam_list(i)/(max(size(Y)));
        param.gamma = gam_list(j)/(max(size(Y)));
        param.init_rank = rank_list(j);
        [L_cp,S_cp,times] = cp_based(Y, param);
        %%%%BATF
        sparse_tensor = Y;
        sparse_tensor(param.ind_m) = 0;
        [model] = BATF_VB(Y,sparse_tensor,'CP_rank',param.init_rank,'maxiter',param.max_iter);
        L_batf = model.tensorHat;
        S_batf = Y-model.tensorHat;
        %%%%
        [L_lrt, S_lrt, ~] = loss(Y, param);
%         param.psi = min(std_m)*std_m.^-2;
%         [L_logss,S_logss,~] = logss(Y, param);
%         param.psi = [0.1, 1,5,0.01];%min(std_m)*std_m.^-1;
        param.lambda = lam_list(i)/(numel(Y));
        param.gamma = gam_list(j)/(numel(Y));
        [L_gloss, S_gloss,~] = gloss_3(Y, param);

%         results_real_nyc(S_gloss, S_lrt, S_whorpca, S_horpca, Y, a_L, OCS, ind_removed, param, true)
        m_gloss = mahal_dist(S_gloss);
        m_loss = mahal_dist(S_lrt);
%         m_logss = mahal_dist(S_logss);
        o_gloss = one_class_svm(S_gloss, out_fr);
        lof_gloss = apply_lof(S_gloss, 10);
        for k=1:len_k
            det_gloss(:,i,j,k) = (detect_real_events((m_gloss), ind_removed, list_k(k)));
            det_gloss_svm(:,i,j,k) = (detect_real_events((o_gloss), ind_removed, list_k(k)));
            det_gloss_lof(:,i,j,k) = (detect_real_events((lof_gloss), ind_removed, list_k(k)));
%             det_logss(:,i,j,k) = (detect_real_events((m_logss), ind_removed, list_k(k)));
            det_cp(:,i,j,k) = (detect_real_events((S_cp), ind_removed, list_k(k)));
            det_batf(:,i,j,k) = (detect_real_events((S_batf), ind_removed, list_k(k)));
            det_loss(:,i,j,k) = (detect_real_events((m_loss), ind_removed, list_k(k)));
            det_whorpca(:,i,j,k) = (detect_real_events((m_whorpca), ind_removed, list_k(k)));
            det_horpca(:,i,j,k) = (detect_real_events((m_horpca), ind_removed, list_k(k)));
        end
    end
end


