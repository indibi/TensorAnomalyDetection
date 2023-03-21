

ind_remain = setdiff(1:numel(Yn), ind_removed);
eps_list = 1.3*10.^[2];
for i = 1:length(eps_list)
    db_Y = apply_dbscan(Yn_d, eps_list(i), 5);
    true_pos = sum(X(db_Y(ind_remain)==1));
    false_neg = sum(X(ind_remain),'all')-true_pos;
    false_pos = sum(db_Y(ind_remain))-true_pos;
    true_neg = sum(1-X(ind_remain),'all')-false_pos;
    precision_dbscan(i,ind_outer) = true_pos/(true_pos+false_pos);
    recall_dbscan(i,ind_outer) = true_pos/(true_pos+false_neg);
    fpr_dbscan(i,ind_outer) = false_pos/(false_pos+true_neg);
end