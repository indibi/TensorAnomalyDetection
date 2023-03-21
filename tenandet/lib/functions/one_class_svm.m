function [scr] = one_class_svm(Y, out_fr)
% 
% Returns results of one-class SVM applied on third mode fibers of four
% mode tensor Y. 

sz_1 = size(Y,1);
sz_2 = size(Y,2);
sz_3 = size(Y,3);
sz_4 = size(Y,4);

scr = zeros(size(Y));
for i=1:sz_1
    for j=1:sz_2
        for k=1:sz_4
            model = fitcsvm(squeeze(Y(i,j,:,k)),ones(sz_3,1),...
            'Standardize',true,'OutlierFraction', out_fr);
            [~,scr(i,j,:,k)] = kfoldPredict(crossval(model));
        end
    end
end
    
scr = -scr;
end