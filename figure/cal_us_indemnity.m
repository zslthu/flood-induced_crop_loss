function [plotdata_group_best, plotdata_group_best_year] = cal_us_indemnity(f_result, loss_type)

% %
name_crop_mod = {'maize','soy','wheat'};
name_climate  = {'wfdei.gpcc','agmerra'};
name_model    = {'cgms_wofost','lpj_guess','clm_crop','lpjml','epic_iiasa','gepic','orchidee_crop','pdssat','papsim','pegasus'};


% % load map data
dir_shape = '../../data/states_contiguous/states_contiguous.shp';
USmap = shaperead(dir_shape);
numericStateFIPS = str2double({USmap.STATE_FIPS});
USmap_table = [table(numericStateFIPS', 'VariableNames', {'state_id'}), ...
    cell2table({USmap.STATE_ABBR}', 'VariableNames', {'STATE_ABBR'})];


%% MAIN CALCULATION
for iv = 1:3%:length(name_v)
    crop_mod = name_mod{iv};

    % %  load result
    for icc = 1:2
        climate  = name_climate{icc};

        dir_best   = [f_result '/' crop_mod '_' loss_type '_noirr_' climate '_fips.csv'];
        data_best  = readtable(dir_best);

        data_best(data_best.time==2010,:)=[];
        data_best.Prec_sigma_bin = [];
        comName = {'time' 'lon' 'lat' 'indemnity' 'FIPS' 'state_id'};
        data_best = data_best(:,[comName  name_model]);
        data_best.indemnity(isnan(data_best.indemnity)) = 0;

        if icc == 1
            mergedTable_best = data_best;
        else
            mergedTable_best = innerjoin(mergedTable_best, data_best, 'Keys', comName);
        end
    end       

    % % calculate average value at each state
    group_best = cal_State_mean(data_best,name_model);
    group_best = innerjoin(group_best, USmap_table, 'Keys', {'state_id'});
    % group_best = sortrows(group_best,'model_ensemble','descend');
    group_best = sortrows(group_best,'sum_indemnity','descend');

    plotdata_group_best.(crop_mod)= group_best;

    % % calculate average value at each year
    group_best_year = cal_Year_mean(data_best,name_model);
    group_best_year.sum_indemnity(group_best_year.sum_indemnity==0)=nan;
    group_best_year.model_ensemble(group_best_year.model_ensemble==0)=nan;
    group_best_year = rmmissing(group_best_year);

    plotdata_group_best_year.(crop_mod) = group_best_year;
end

end

%% functions
function group_best = cal_State_mean(data_best,name_model)

data_best = fillmissing(data_best, 'constant', 0);
group_best = groupsummary(data_best, 'state_id', 'sum', 'indemnity');
calsum = @(values) sum(values,'omitnan');

for im = 1:length(name_model)
    eval(['group_best.sum_' name_model{im} ' = splitapply(calsum, data_best.' name_model{im} ', findgroups(data_best.state_id));'])
end
group_best.model_ensemble = median(group_best{:,4:13},2);

end


function group_best = cal_Year_mean(data_best,name_model)

data_best = fillmissing(data_best, 'constant', 0);
group_best = groupsummary(data_best, 'time', 'sum', 'indemnity');
calsum = @(values) sum(values,'omitnan');

for im = 1:length(name_model)
    eval(['group_best.sum_' name_model{im} ' = splitapply(calsum, data_best.' name_model{im} ', findgroups(data_best.time));'])
end
group_best.model_ensemble = median(group_best{:,4:13},2);

end


