%% Outlying score Experiment 1: Varying amplitude of anomaly vs separability
% 
%
%% Set Project Path
clear all
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

%% Load NYC data 
sizes = [144,7,52,10];                                      
load nyc_tensors.mat
load regions.mat
arrs = squeeze(sum(reshape(arrs,6,24,365,[]),1)); 
Y = double(reshape(arrs(:,1:364,regions), 24, 7, 52,[]));% Select 81 zones
Y(:,1,53,:) = arrs(:,365,regions);
% Fill in the last week with the mean of other weeks
Y(:,2:7,53,:) = mean(Y(:,2:7,1:52,:),3);
sz = size(Y);
%% Experiment variables
% Control variables     -------
NoAD = 5;           % Number of anomalous locations
NoA = 100*NoAD;     % Total number of anomalies 
NoT = 1;            % Number of trials
LoA = 8;            % Length of anomalies
% Independent variables -------
cs = [2,3,4,5,10];  % Range of anomaly amplitudes
% Dependent variable    -------
os = zeros(81,length(cs),NoT);% Outlying scores

%% 
% 
select_locs = cell(length(cs),1);
Anomaly_masks = cell(length(cs),1);
anomaly_masks = cell(length(cs),1);
as = cell(length(cs),1);
Yns = cell(length(cs),1);
Ls = cell(length(cs),1);
Ss =  cell(length(cs),1);
params = cell(length(cs),1);
tics = zeros(length(cs),1);
tocs = zeros(length(cs),1);
Sns = cell(81,length(cs));
zs = cell(length(cs),1);

for i=1:length(cs)
    for t=1:NoT
    % Create anomalies ------------
    select_locs{i} = randperm(81,NoAD);
    Anomaly_masks{i} = zeros(sz);
    as{i} = zeros([sz(1:3),5]);
    [as{i}, anomaly_masks{i}] = add_persistent_anomaly(Y(:,:,:,select_locs{i}),LoA,NoA,cs(i),1);
    Yns{i} = Y;
    Yns{i}(:,:,:,select_locs{i}) = Y(:,:,:,select_locs{i}) + as{i};
    Anomaly_masks{i}(:,:,:,select_locs{i}) = anomaly_masks{i};
    
    % Run LOSS ------------------
    params{i} = set_params(Yns{i});
    params{i}.lambda=5e-2;
    params{i}.gamma=5e-2;
    params{i}.err_tol=1e-4;
    params{i}.max_iter=300;

    [Ls{i},Ss{i},~,~]= loss(Yns{i}, params{i});

    
    % Calculate outlying score ----
    Sn = cell(81,1);
    for n =1:81
        Sn{n}= squeeze(Ss{i}(:,:,:,n));
    end
    for n=1:81
        z = Sn{n}; 
        [os(n,i,t), ~]= outlying_function2(z,Sn,100,1e-3,0);
    end
    
    labels= {'Normal'};
    labels =repmat(labels,81,1);
    for k=select_locs{i}
        labels{k} = 'Anomaly'; 
    end
    classNames = cell(2,1);
    classNames{1}='Normal';
    classNames{2}='Anomaly';
    score = [-os(:,i,t),os(:,i,t)];
    rocObj = rocmetrics(labels,score,classNames);
    auc(i,t) = rocObj.AUC(1);
    end
end



