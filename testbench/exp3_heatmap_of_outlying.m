clear % Add repository folders to path
mfilePath = mfilename('fullpath');
if contains(mfilePath,'LiveEditorEvaluationHelper')
    mfilePath = matlab.desktop.editor.getActiveFilename;
end
folder_path = strsplit(mfilePath,filesep);
repo_path = join([folder_path(1:end-2)],filesep);
repo_path = repo_path{1};   
data_path = [repo_path,filesep,'data'];
src_path = [repo_path,filesep,'src'];
tenandet_path = [repo_path,filesep,'tenandet'];
testbench_path = [repo_path,filesep,'testbench'];
addpath(genpath(data_path), genpath(src_path),genpath(tenandet_path), genpath(testbench_path));
clear folder_path mfilePath repo_path data_path testbench_path src_path tenandet_path

%% LOAD NYC DATA
sizes = [144,7,52,10];                                      % Load NYC data 
load nyc_tensors.mat
load regions.mat
arrs = squeeze(sum(reshape(arrs,6,24,365,[]),1)); 
Y = double(reshape(arrs(:,1:364,regions), 24, 7, 52,[]));   % Select 81 zones
Y(:,1,53,:) = arrs(:,365,regions);
Y(:,2:7,53,:) = mean(Y(:,2:7,1:52,:),3);                 % Fill in the last week with the mean of other weeks
sz = size(Y);                               
%% SET UP EXPERIMENT
% Control variables
c = 4;
NoAL = 5;
NoA = 100*NoAL;
LoA =8;
NoT = 5;
loss_maxit = 250;
of_maxit = 30;
of_errtol = 1e-3;
% Bookkeeping
len_lda = 8;
len_gma = 6;
ldas = logspace(0,-4,len_lda);
gammas = logspace(0,-4,len_gma);
outlying_scores = zeros(len_lda*len_gma,NoT,81);
auc = zeros(len_lda*len_gma,NoT);
[Ldas,Gammas] = meshgrid(ldas,gammas);
hyper_params = [Ldas(:), Gammas(:)];
[Lda_idxs,Gamma_idxs] = meshgrid(1:len_lda,1:len_gma);
idxs = [Lda_idxs(:), Gamma_idxs(:)];
% Randomly select anomalous locations (constant in the experiment)
select_loc = randperm(81,NoAL);
%% RUN EXPERIMENT IN PARALLEL
delete(gcp('nocreate'));
parpool(len_lda*len_gma);
tstart = tic;
parfor i=1:len_lda*len_gma
    lda = hyper_params(i,1);
    gamma = hyper_params(i,2);
    for j=1:NoT
        % Generate random anomaly and apply it
        Anomaly_mask = zeros(sz);
        a = zeros([sz(1:3),5]);
        [a, anomaly_mask] = add_persistent_anomaly(Y(:,:,:,select_loc),LoA,NoA,c,1);
        Yn = Y;
        Yn(:,:,:,select_loc) = Y(:,:,:,select_loc) + a;
        Anomaly_mask(:,:,:,select_loc) = anomaly_mask;

        % Apply LOSS algorithm
        param = set_params(Yn);
        param.lambda=lda;
        param.gamma=gamma;
        param.err_tol=1e-4;
        param.max_iter=loss_maxit;
        [L,S]= loss(Yn, param);
        
        % Calculate Outlying function
        Sn = cell(81,1);
        outlying_score = zeros(81,1);
        for n =1:81
            Sn{n}= squeeze(S(:,:,:,n));
        end

        for n=1:81
            z = Sn{n};
            [OS, Or]= outlying_function2(z,Sn,of_maxit,1e-3,0);
            outlying_score(n)= OS;
            if mod(n,27)==0
                disp(n)
            end
        end
        outlying_scores(i,j,:)= outlying_score;

%         label= {'N'};
%         label =repmat(label,81,1);
%         label(select_loc) = repmat({'A'},[NoAL,1]);
%         classNames = cell(2,1);
%         classNames{1}='N';
%         classNames{2}='A';
%         score = [-squeeze(outlying_scores(i,j,:)),squeeze(outlying_scores(i,j,:))];
%         try
%             rocObj = rocmetrics(label,score,classNames);
%             auc(i,j) = rocObj.AUC(1);
%         catch ME
%             auc(i,j) = NaN;
%             disp('Sacma sapan bisi')
%         end
    end
end
tEnd = toc(tstart);

%% SAVE RESULTS
% results.auc = zeros(len_lda,len_gma,NoT);
results.outlying_scores = zeros(len_lda,len_gma,NoT,81);
for i = 1:len_lda*len_gma
    lda_idx = idxs(i,1);
    gamma_idx = idxs(i,2);
%     results.auc(lda_idx,gamma_idx,:) = auc(i,:);
    results.outlying_scores(lda_idx,gamma_idx,:,:) = outlying_scores(i,:,:);
end

results.select_loc = select_loc;
results.ldas = ldas;
results.gammas = gammas;
results.cs = 4;
results.NoAL = NoAL;
results.NoA = NoA;
results.LoA = LoA;
results.NoT = NoT;
results.time_spent = tEnd;
i = 1;
formatSpec = '%i';
fname = 'exp3_HPheatmap_auc';
while isfile([fname,num2str(i,formatSpec),'.mat'])
    i = i+1;
end
save([fname,num2str(i,formatSpec),'.mat'],'results');
%% Notify with sound
%load handel
%sound(y,Fs)
%clear y Fs