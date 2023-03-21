function [X] = create_labels(data)
%% [X] = createLabels(data)
%   Creates anomaly labels using the information provided in the struct
%   data
% Parameters:
%  data: Struct with fields:
%      station_counts: Matrix with rows corresponding to all year with 5
%      minute intervals, and columns corresponding to sensors, or arrival
%      and departure information.
%      station_ids: Zone or bike station id.
%      station_times: Start times of all trips.
%      inc_timestamp: Start times for the anomalous target events.
%      inc_duration: Duration of the target events:
%      inc_station_ids: Stations or zones at which anomalous event occurs.
num_elements = size(data.station_counts,1);
num_sensors  = size(data.station_counts,2);
X = zeros(num_elements, num_sensors);
for i=1:length(data.station_ids)
    ind_start = round(minutes(data.inc_timestamp(i,:)-...
        datetime(data.station_times(1,:)))/5);
    ind_end = round(data.inc_duration(i)/5)+ind_start;
    [~,ind_station,~] = intersect(data.station_ids,data.inc_station_id(i));
    X(ind_start:ind_end, ind_station) = 1;
end
end