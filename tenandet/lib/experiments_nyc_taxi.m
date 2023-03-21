clear
sizes = [144,7,52,10];
load nyc_tensors.mat
load regions.mat
arrs = squeeze(sum(reshape(arrs,6,24,365,[]),1));
Y = double(reshape(arrs(:,1:364,regions), 24, 7, 52,[]));
Y(:,1,53,:) = arrs(:,365,regions);
Y(:,2:7,53,:) = mean(Y(:,2:7,1:52,:),3);
ind_removed = [];
for i=1:4
    std_m(i) = sum(std(t2m(Y,i)));
end

list_k = [100,500,1000,2000,5000,7000,14000,21000];
lam_list = 10.^[3];
gam_list = 10.^[-3];
rank_list = 11;
len_l = length(lam_list);
len_g = length(gam_list);
len_k = length(list_k);
det_gloss = zeros(20,length(lam_list),len_g,len_k);
det_gloss_lof = det_gloss;
det_gloss_svm = det_gloss;
det_loss = det_gloss;
det_logss = det_gloss;
det_whorpca = det_loss;
det_horpca = det_loss;
det_ee = det_loss;
det_lof = det_loss;
det_ocsvm = det_loss;
det_cp = det_ee;
det_batf = det_ee;

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
    param.lambda = lam_list(i)/sqrt(max(size(Y)));
    param.alpha = 0;
    param.disp = false;
    param.psi = [1,1,1,1];
    param.beta_1 = 1/(5*std(Y(:)));
    param.max_iter = 100;
    param.err_tol = 0.01;
    tic;
    [~,S_horpca, ~] = horpca(Y, param);
    time_h(i) = toc;
    param.psi = max(std_m)*std_m.^-1;
    tic;
    [~,S_whorpca, ~] = horpca(Y, param);
    time_wh(i) = toc;
    
    m_S_whorpca = mahal_dist(S_whorpca);
    m_S_horpca = mahal_dist(S_horpca);
    for j=1:len_g
        param = [];
        param.ind_m = ind_removed;
        param.psi = max(std_m)*std_m.^-1;
        param.alpha = 0;
        param.disp = false;
        param.beta_1 = 1/(5*std(Y(:)));
        param.beta_2 = param.beta_1;
        param.beta_3 = param.beta_1;
        param.beta_4 = param.beta_1;
        param.beta_5 = param.beta_1;
        param.max_iter = 100;
        param.err_tol = 0.01;
        param.lambda = lam_list(i)/(max(size(Y)));
        param.gamma = gam_list(j)/(max(size(Y)));
        param.mu = 10^-3;
        param.init_rank = rank_list(j);
        tic;
        [L_cp,S_cp,times] = cp_based(Y, param);
        time_cp{j}(i) = toc;
        %%%%BATF
        sparse_tensor = Y;
        sparse_tensor(param.ind_m) = 0;
        tic;
        [model] = BATF_VB(Y,sparse_tensor,'CP_rank',param.init_rank,'maxiter',param.max_iter);
        time_batf{j}(i) = toc;
        L_batf = model.tensorHat;
        S_batf = Y-model.tensorHat;
        S_batf = mahal_dist(S_batf);
        %%%%
        param.lambda = 10/(max(size(Y)));
        param.gamma = 10^6/(max(size(Y)));
        tic;
        [~, S_loss, ~] = loss(Y, param);
        time_loss{j}(i) = toc;
        % param.lambda = 1/sqrt(max(size(Y)));
        param.lambda = lam_list(i)/(numel(Y));
        param.gamma = gam_list(j)/(numel(Y));
        param.theta = 1/(sum(std(t2m(Y,4),[],2).^2))^2;
        tic;
        [L_gloss,S_gloss,~] = gloss_3(Y, param);
        time_g{j}(i) = toc;
%         param.lambda = lam_list(i)/(max(size(Y)));
%         param.gamma = gam_list(j)/max(size(Y));
%         param.theta = 10^-1/geomean(std_m);
%         tic;
%         [~,S_logss,~] = logss(Y, param);
%         time_l(i,j) = toc;

        % results_real_nyc(S_gloss_3, S_lrt, S_whorpca, S_horpca, Y, a_L, OCS, ind_removed, param, true)
        
        
        a_L = apply_lof(S_gloss, 10);
        OCS = one_class_svm(S_gloss, out_fr);
        m_S_gloss = mahal_dist(S_gloss);
%         m_S_logss = mahal_dist(S_logss);
        m_S_loss = mahal_dist(S_loss);
        for k=1:len_k
            det_gloss(:,i,j,k) = (detect_real_events(m_S_gloss, ind_removed, list_k(k)));
            det_gloss_lof(:,i,j,k) = (detect_real_events(a_L, ind_removed, list_k(k)));
            det_gloss_svm(:,i,j,k) = (detect_real_events(OCS, ind_removed, list_k(k)));
            det_cp(:,i,j,k) = (detect_real_events((S_cp), ind_removed, list_k(k)));
            det_batf(:,i,j,k) = (detect_real_events((S_batf), ind_removed, list_k(k)));
%             det_logss(:,i,j,k) = (detect_real_events(m_S_logss, ind_removed, list_k(k)));
            det_loss(:,i,j,k) = (detect_real_events(m_S_loss, ind_removed, list_k(k)));
            det_horpca(:,i,j,k) = (detect_real_events(m_S_horpca, ind_removed, list_k(k)));
            det_whorpca(:,i,j,k) = (detect_real_events(m_S_whorpca, ind_removed, list_k(k)));
        end
    end
end

