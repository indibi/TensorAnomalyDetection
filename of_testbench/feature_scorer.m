function [anomaly_score] = feature_scorer(outlying_score,S, normalize)

anomaly_score = zeros(size(S));
sabs = abs(S);
if normalize
    for l = 1:81
        for w = 1:53
        anomaly_score(:,:,w,l) = outlying_score(w,l)*sabs(:,:,w,l)/max(sabs(:,:,w,l),[],'all');
        end
    end
else
    for l = 1:81
        for w = 1:53
        anomaly_score(:,:,w,l) = outlying_score(w,l)*sabs(:,:,w,l);
        end
    end
end

