function [db_Y] = apply_dbscan(Y, eps_1, k)
% [db_Y] = apply_dbscan(Y, eps_1, k)
% Function that applies st-DBSCAN to all data.

% eps_2 = 1;
% delta_e = 1.2;
db_Y = zeros(size(Y));
for l = 1:size(Y, 4)
    for i = 1:size(Y, 1)
        for j = 1:size(Y, 2)
            D{1} = squeeze(Y(i,j,:,l));
            [db_Y(i,j,:,l)] = dbscan(D{1}, eps_1, k);
        end
    end
end
db_Y = db_Y<0;
end