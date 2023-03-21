function [tensor, dates, county_names] = read_covid_csv(filename, varargin)
% [tensor, dates, county_names] = read_covid_csv(filename, varargin)
% reads covid19 data and produces tensor with size: 
%      number_counties x 7(days in a week) x number_weeks
if length(varargin)<2
    numweeks = 17;
else
    numweeks = varargin{2};
end
if length(varargin)<1
    state = 'MI';
else
    state = varargin{1};
end
if nargin ==0
    filename = 'covid_confirmed_usafacts.csv';
end
covid19 = readtable(filename);
mask = cellfun(@contains, table2cell(covid19(:,3)), repmat({state}, size(covid19,1),1));

county_names = table2cell(covid19(mask,2));
county_names = county_names(2:end);
dates = table2cell(covid19(1,5:end));
dates = dates(2:end);

mat = (table2cell(covid19(mask, 5:end)));
mat = cellfun(@str2num, mat);
mat = diff(mat(2:end,3:end),1,2);
% mat = mat(:,49:end); % Clear zeros for many states. 
% dates = dates(49:end);
tensor = permute(reshape(mat(:,1:(numweeks-1)*7),[],7,numweeks-1),[2,3,1]);
% save('covid_19data', 'dates', 'county_names', 'tensor');
end