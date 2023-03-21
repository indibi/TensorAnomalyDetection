function [X, Y, Yn, days, weeks, sensors] = get_traffic_data(filename)
%gendata Generate four-mode simulation tensor for anomaly detection.
%   [X, Y, Yn] = get_traffic_data(filename)
%
%   Outputs:
%   X        : Labels of anomalies. 1 if there is an anomaly, -1 otherwise.
%   Y        : Data with anomalies.
%   Yn       : Noisy anomaly data.
%   days     : Missing days of a week.
%   weeks    : Weeks corresponding to the missing days.
%   sensors  : Sensors with missing days.
%
%   example:
%   [X, Y, Yn, days, weeks, sensors] = get_traffic_data;

if nargin==0
    traffic_data = load('24-Oct-2018_data.mat');
else
    traffic_data = load(filename);
end

X = create_labels(traffic_data);
num_sensors = size(X, ndims(X));
X = reshape(X, 288, 365, num_sensors);
X = reshape(X(:,1:364,:), 288, 7, 52, num_sensors);
Y = reshape(traffic_data.station_counts, 288, 365, num_sensors);
Y = reshape(Y(:,1:364,:), 288, 7, 52, num_sensors);
Y = denan(Y);
dims = size(Y);
rng(123);

num_missing_days = 20;
days = randi(dims(3)*dims(2), num_missing_days, 1);
weeks = floor(days/7)+1;
days = mod(days, 7)+1;
sensors = randi(num_sensors, num_missing_days);
Yn = Y;

for i=1:num_missing_days
    Yn(:,days(i), weeks(i), sensors(i)) = 0;
end
end