% Script for reading NYC Yellow Taxi Trip data for 2018.
clear
days_of_year = {1:31,1:28,1:31,1:30,1:31,1:30,1:31,1:31,1:30,1:31,1:30,1:31};
d_count=1;
deps = zeros(144,365,263,'uint16');
arrs = deps;
for i=1:12
    file = ['\\cifs.egr.msu.edu\research\sigimprg\Emre\databases\nyc_taxi\yellow_tripdata_2018-',num2str(i,'%02d'),'.csv'];
    tab = readtable(file);
    dates = table2array(tab(:,2:3));
    locs = table2array(tab(:,8:9));
    clear tab
    for day_id = days_of_year{i}
        for t=1:144
            dt_lower(t) = datetime(2018,i,day_id,floor((t-1)/6),mod(10*(t-1),60),0);
            dt_upper(t) = datetime(2018,i,day_id,floor(t/6),mod(10*t,60),0);
            mask = isbetween(dates, dt_lower(t), dt_upper(t));
            for loc_id=1:263
                deps(t,d_count,loc_id) = sum(locs(mask(:,1),1)==loc_id);
                arrs(t,d_count,loc_id) = sum(locs(mask(:,2),2)==loc_id);
            end
        end
        d_count = d_count+1;
    end
end
save('data\nyc_tensors.mat','deps','arrs')
