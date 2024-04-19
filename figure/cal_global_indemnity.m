function [loss_compare, loss_compare_crop, perform_ci, data_loss, data_loss_all]=cal_global_indemnity(f_result, loss_type)
warning('off', 'all');

% % % % model information
name_crop_mod = {'maize','soy','wheat'};
name_model    = {'cgms-wofost','lpj-guess','clm-crop','lpjml','epic-iiasa','gepic','orchidee-crop','pdssat','papsim','pegasus'};
name_climate  = {'wfdei.gpcc','agmerra'};


yr_start = 1991;
yr_end   = 2009;
period   = yr_start:yr_end;

%% read report data
dir_report  = '../../data/natural-disasters-M49.csv';
data_report = readtable(dir_report);
idx_f       = find(contains(data_report.Properties.VariableNames,'Flood'));
data_report = data_report(:,[1,2,3,idx_f]);
data_report.TotalEconomicDamagesFromFloods = data_report.TotalEconomicDamagesFromFloods;
idx_year    = data_report.Year>=yr_start & data_report.Year<=yr_end;
data_report = data_report(idx_year,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAIN CALCULATION
%%%%%%%%%
data_loss_temp = table();
for ic = 1:3
    v_crop = name_crop_mod{ic};
    % % % % read loss result
    data_model = table();
    for imm = 1:length(name_model)
        for icc = 1:2
            climate = name_climate{icc};
            dir_output = [f_result '/loss_' loss_type '/loss_' loss_type '_' v_crop '_' name_model{imm} '_' climate '.csv'];
            eval(['data_model_temp_c' num2str(icc) ' = readtable(dir_output);'])
        end
        data_model_temp=data_model_temp_c1;
        data_model_temp{:,4:end} = (data_model_temp_c1{:,4:end}+data_model_temp_c2{:,4:end})/2;
        data_model_temp.Properties.VariableNames{'M49'} = 'Area_Code_M49_';
        data_model_temp.Properties.VariableNames{'CountryName'} = 'Area';
        data_model_temp{:,4:end} = fillmissing(data_model_temp{:,4:end},'constant',0);
        data_model = [data_model;data_model_temp];
    end
    summaryTable_temp = groupsummary(data_model,{'Area_Code_M49_','Area','Item'}, 'mean');
    data_loss_temp = [data_loss_temp;summaryTable_temp];
end
data_loss_all = data_loss_temp;
data_loss_all(:,{'GroupCount'})=[];

data_loss_temp(:,{'Item'})=[];
data_loss = groupsummary(data_loss_temp,{'Area_Code_M49_','Area'}, 'sum');
data_loss(:,{'GroupCount','sum_GroupCount'})=[];
for iy = 1:length(period)
    data_loss.Properties.VariableNames{['sum_mean_Y' num2str(period(iy))]} = ['Y' num2str(period(iy))];
end
%%%%%%%%%

loss_mod = table((yr_start:yr_end)', 'VariableNames', {'Year'});
area_code = 0; % 'global'-0 'china'-159 'USA'-840 'UK'-628  'Australia'-36 'India'-356
[loss_compare, perform_ci]  = cal_perform(data_report,data_loss,loss_mod,area_code);

for ic=1:3
    data_loss_crop = data_loss_all(strcmp(data_loss_all.Item,name_crop_mod(ic)),:);
    data_loss_crop.Item = [];
    loss_mod = table((yr_start:yr_end)', 'VariableNames', {'Year'});
    area_code = 0;
    [loss_compare_crop{ic}, perform_ci] = cal_perform(data_report, data_loss_crop,loss_mod,area_code);
end


end

%% functions
function [loss_compare, perform_idx] = cal_perform(data_report,data_loss, loss_mod, area_code)

idx_y    = find(contains(data_loss.Properties.VariableNames,'Y')); idx_y = idx_y(1); 
loss_mod.loss_mod = data_loss{data_loss.Area_Code_M49_==area_code,idx_y:end}';

idx_world = data_report.M49 == area_code;
loss_obs = data_report(idx_world,{'Year','TotalEconomicDamagesFromFloods'});

loss_compare = outerjoin(loss_mod,loss_obs,'Keys','Year','MergeKeys', true);
loss_compare = rmmissing(loss_compare); 
loss_compare.Properties.VariableNames = {'Year','loss_mod','loss_obs'};

if sum(loss_mod.loss_mod)~=0 && size(loss_compare,1)>=10
    [~,loss_mod_sort_idx] = sort(loss_compare.loss_mod,'descend');
    loss_mod_sort_idx = loss_mod_sort_idx(1:round(end/2));
 
    [~,loss_obs_sort_idx] = sort(loss_compare.loss_obs,'descend');
    loss_obs_sort_idx = loss_obs_sort_idx(1:round(end/2));

    perform_idx = length(intersect(loss_obs_sort_idx,loss_mod_sort_idx))/length(loss_obs_sort_idx);
else
    perform_idx = nan;
end
end

