function Y = denan(Y)
% Y = denan(Y)
% Clears nan values by taking the mean of the rest of the data with no nans
% at those indices.
modes = 1:2;
[~, c] = find(isnan(t2m(Y, modes)));
sensors_with_nans = floor(c/52)+1;
sensor_inds = unique(sensors_with_nans);
for j = 1:length(sensor_inds)
    weeks = unique(c(sensors_with_nans==sensor_inds(j)));
    for jj = 1:length(weeks)
        [~, day_with_nans] = find(isnan(squeeze(Y(:,:,mod(weeks(jj), 52),sensor_inds(j)))));
        day_inds = unique(day_with_nans);
        for i = 1:length(day_inds)
            sig = Y(:,day_inds(i),mod(weeks(jj),52),sensor_inds(j));
            inds = find(isnan(sig));
            nonans = setdiff(1:52, mod(weeks(jj),52));
            sig(inds) = mean(Y(inds,day_inds(i), nonans,sensor_inds(j)), 3);
            Y(:,day_inds(i),mod(weeks(jj),52),sensor_inds(j)) = sig;
        end
    end
end
end