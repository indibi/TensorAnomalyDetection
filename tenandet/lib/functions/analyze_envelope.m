function [fpr, recall] = analyze_envelope(S, X, ind_removed)

alpha = [.1:.2:10];

dims = size(S);
if isempty(ind_removed)
    removs=[0,0,0];
else
    [~,remov_i,remov_j,remov_k] = ind2sub(dims, ind_removed);
    removs = unique([remov_i,remov_j,remov_k],'rows');
end

m_S = mean(S, 3);
st_S = std(S, [], 3);
est_anoms = zeros(dims);
fpr = zeros(length(alpha),1);
recall = zeros(length(alpha),1);
d2 = dims(2);
d3 = dims(3);
for r = 1:length(alpha)
    for j=1:dims(4)
        for i=1:d2
%             weeks = 1:d3;
%             [ts_removed, ~, inds] = intersect([i,j], ...
%                 removs(:,[1,3]),'rows');
%             if ~isempty(ts_removed)
%                 weeks_removed = removs(inds, 2);
%                 weeks = setdiff(weeks, weeks_removed);
%             end
            upper_env = squeeze(m_S(:,i,j)+alpha(r)*st_S(:,i,j));
            lower_env = squeeze(m_S(:,i,j)-alpha(r)*st_S(:,i,j));
            for k=1:d3
                if isempty(intersect([i,k,j], removs, 'rows'))
                    est_anoms(:,i,k,j) = S(:,i,k,j)>upper_env |...
                        S(:,i,k,j)<lower_env;
                end
            end
        end
    end
    recall(r) = sum(X(est_anoms==1)==1,'all')/sum(X,'all');
    fpr(r) = sum(X(est_anoms==1)==0,'all')/sum(X==0,'all');
end

end