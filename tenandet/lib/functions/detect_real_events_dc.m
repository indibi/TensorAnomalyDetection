function detected = detect_real_events_dc(S, mask, num_k)
%% Event Detection For DC Bike Data

if nargin==2
    num_k = 7000;
end
load('washington_dates.mat', 'dates');
file = '\\cifs.egr.msu.edu\research\sigimprg\Emre\databases\bike\washington\DC_Arl_census.shp';
S2 = shaperead(file);
cnss_tract = zeros(1,length(S2));
for i=1:length(S2)
    cnss_tract(i) = str2double(S2(i).NAMELSAD(14:end));
end
num_anom = 20;

[~, ind] = sort(abs(S(:)),'descend');
ind = ind_rem(ind);
[i_1, i_2, i_3, i_4] = ind2sub(size(S), ind(1:num_k)');

detected = zeros(1,num_anom);
for i=1:num_anom
    [s_h, s_d, s_w] = date2ind(dates{1,i}(1));
    [e_h, e_d, e_w] = date2ind(dates{1,i}(2));
    if e_d~=s_d || s_w~=e_w
        error('Check dates!')
    end
    det_hours = s_h:e_h;
    for j = 1:length(dates{3,i})
        roi = repmat([s_d, s_w, find(find(mask)==(dates{3,i}(j)))], length(det_hours),1);
        det_inds = [det_hours', roi];
        if ~isempty(intersect(det_inds, [i_1, i_2, i_3, i_4],'rows'))
            detected(i) = true;
            break;
        end
    end
end
end
