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
cs = [2,3,4,5,6]; % ANOMALY AMPLITUDE
NoAD = 5;         % Number of anomalous locations
NoA = 100*NoAD;   % Total number of anomalies ~100 anomalies per location
LoA =8;           % Length of the anomalies
NoT = 10;         % Number of trials (Experiment repeat)
len_cs = length(cs); 
outlying_scores = zeros(NoT,length(cs),81);
auc = zeros(NoT,len_cs);
%%
delete(gcp('nocreate'));
parpool('local',NoT);
tstart = tic;
parfor i=1:NoT
    for j =1:len_cs
        select_loc = randperm(81,NoAD);
        Anomaly_mask = zeros(sz);
        a = zeros([sz(1:3),5]);
        [a, anomaly_mask] = add_persistent_anomaly(Y(:,:,:,select_loc),LoA,NoA,cs(j),1);
        Yn = Y;
        Yn(:,:,:,select_loc) = Y(:,:,:,select_loc) + a;
        Anomaly_mask(:,:,:,select_loc) = anomaly_mask;
        
        Sn = cell(81,1);
        outlying_score = zeros(81,1);
        for n =1:81
            Sn{n}= squeeze(Yn(:,:,:,n));
        end

        for n=1:81
            z = Sn{n};
            %outlying_score(n) 
            [OS, Or]= outlying_function2(z,Sn,30,1e-3,0);
            outlying_score(n)= OS;
            if mod(n,10)==0
                disp(n)%outlying_score(n)
            end
        end
        outlying_scores(i,j,:)= outlying_score;

        labels= {'Normal'};
        labels =repmat(labels,81,1);
        for k=select_loc
            labels{k} = 'Anomaly'; 
        end
        classNames = cell(2,1);
        classNames{1}='Normal';
        classNames{2}='Anomaly';
        score = [-outlying_scores(i,j,:),outlying_scores(i,j,:)];
        try
            rocObj = rocmetrics(labels,score,classNames);
            auc(i,j) = rocObj.AUC(1);
        catch ME
            disp('Sacma sapan bisi')
        end
    end
end

%% SAVE RESULTS
results.auc = auc;
results.cs = cs;
results.NoAD = NoAD;
results.NoA = 100*NoAD;
results.LoA = LoA;
results.NoT = NoT;
results.time_spent = tEnd;
save('exp1_amplitude_vs_auc_ez3.mat','results');
%% Notify with sound
load handel
sound(y,Fs)
clear y Fs