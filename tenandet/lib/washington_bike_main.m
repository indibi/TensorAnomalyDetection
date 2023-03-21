clear
sizes = [144,7,52,10];
load washington_bikedata.mat
arrs = squeeze(sum(reshape(arrs,6,24,365,[]),1));
mask = sum(t2m(arrs,3),2)>0;
Y = double(reshape(arrs(:,1:364,mask), 24, 7, 52,[]));
Y(:,1,53,:) = arrs(:,365,mask);
Y(:,2:7,53,:) = mean(Y(:,2:7,1:52,:),3);
ind_removed = [];
for i=1:4
    std_m(i) = sum(std(t2m(Y,i),[],1));
end



list_k = [20,100,500,1000,2000,5000,7000,14000,21000,30000];
lam_list = 10.^[5];
gam_list = 10.^[-2];
len_g = length(gam_list);
len_k = length(list_k);
det_gloss = zeros(length(lam_list),len_g,len_k);
det_lrt = det_gloss;
det_whorpca = det_lrt;
for i=1:length(lam_list)
    for j=1:len_g
        param = [];
        param.ind_m = ind_removed;
        param.lambda = lam_list(i)*1/(max(size(Y)));
        param.gamma = gam_list(j)/(max(size(Y)));
        param.theta = 1/(sum(std(t2m(Y,4),[],2).^2))^2;
        param.alpha = 0;
        param.psi = max(std_m)*std_m.^-1;
        param.beta_1 = 1/(5*std(Y(:)));
        param.beta_2 = param.beta_1;
        param.beta_3 = param.beta_1;
        param.beta_4 = param.beta_1;
        param.beta_5 = param.beta_1;
        param.max_iter = 100;
        param.err_tol = 0.01;
        [~, S_lrt, ~] = low_temp_sp_dec(Y, param);
        % param.lambda = 1/sqrt(max(size(Y)));
        [~,S_whorpca, ~] = horpca(Y, param);
        % param.psi = [1,1,1,1];
        % [~,S_horpca, ~] = horpca(Y, param);
        param.lambda = lam_list(i)/(numel(Y));
        param.gamma = gam_list(j)/(numel(Y));
        [~,S_gloss_3,~] = gloss_3(Y, param);
        % a_L = apply_lof(Y, 10);
        % out_fr = 0.1;
        % OCS = one_class_svm(Y, out_fr);

        % results_real_nyc(S_gloss_3, S_lrt, S_whorpca, S_horpca, Y, a_L, OCS, ind_removed, param, true)
        
        for k=1:len_k
            det_gloss(i,j,k) = sum(detect_real_events(mahal_dist(S_gloss_3), ind_removed, list_k(k)));
            det_lrt(i,j,k) = sum(detect_real_events(mahal_dist(S_lrt), ind_removed, list_k(k)));
            det_whorpca(i,j,k) = sum(detect_real_events(mahal_dist(S_whorpca), ind_removed, list_k(k)));
        end
    end
end

