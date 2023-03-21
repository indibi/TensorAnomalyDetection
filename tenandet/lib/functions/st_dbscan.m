function [D, lbl] = st_dbscan(D, eps_1, eps_2, delta_e, k)
% [D, lbl] = st_dbscan(D, eps_1, eps_2, delta_e, k)
% Function that applies st-DBSCAN clustering on spatiotemporal data.
% Inputs:
%   D : Data of three cells.
%       D{1} : Matrix with spatial location or vector on rows.
%       D{2} : Matrix with temporal vector on rows.
% Outputs:
%   D : Data with cell corresponding to clusters filled.
%       D{3} : Label vector, returned full in output.
%   lbl : Noise label, 1 if noise, or outlier.

clusterLabel = 0;
for i=1:size(D{1},1)                                                 %(i)
    if D{3}(i) == 0                                                  %(ii)
        X = retrieveNeighbors(D, i, eps_1, eps_2, 0);                %(iii)
        if length(X) < k 
            D{3}(i) = -1;                                            %(iv)
        else                                                         % construct a new cluster(v)
            clusterLabel = clusterLabel + 1;
            clusterItem = D{2}(i,:);
            D{3}(i) = clusterLabel;
            queue = i;                                               %(vi)
            while isempty(queue) == 0
                ptCurrent = queue(1);  
                queue(1) = [];        
                Y = retrieveNeighbors(D, ptCurrent, eps_1, eps_2, clusterLabel);
                if length(Y) >= k
                    for j=1:length(Y)                                %(vii)
                        % |Cluster_Ave()-o.value|<e
                        if D{3}(Y(j))>=0 && norm(mean(clusterItem,1)-D{2}(Y(j),:))<delta_e
                            D{3}(Y(j)) = clusterLabel;               % mark o with current cluster label
                            clusterItem = [clusterItem; D{3}(Y(j), :)];
                            queue = [queue, Y(j)];                   % push
                        end
                    end
                else 
                    D{3}(ptCurrent) = clusterLabel;
                    clusterItem = [clusterItem; D{2}(ptCurrent, :)];
                end
            end
        end
    end
end
lbl = D{3}<0;
end