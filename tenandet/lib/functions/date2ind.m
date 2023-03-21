function [hour, day, week] = date2ind(date)
% Converts the date information to hours, days and weeks for 2018.
temp = duration(date-datetime(2018,01,01,00,00,00));
hour_t = mod(hours(temp),24); 
hour = floor(hour_t)+1;    
temp = days(temp);
day = ceil(mod(temp, 7));
if hour_t==0
    day = day+1;
end
if day==0
    day = 1;
end
week = floor(temp/7)+1;
end