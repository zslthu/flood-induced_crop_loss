clc;clear;

name_obs     = {'corn','soybeans','wheat'};
name_mod     = {'maize','soy','wheat'};
name_climate = {'wfdei.gpcc','agmerra'};
name_title     = {'maize','soybean','wheat'};

% % file path
f_result = '../../output/';

% % load map data for spatial pattern
dir_shape = '../../data/counties_contiguous/counties_contiguous.shp';
USmap = shaperead(dir_shape);
dir_shape = '../../data/states_contiguous/states_contiguous.shp';
USmap_states = shaperead(dir_shape);
dir_tif  = '../../data/counties_contiguous/USA_005.tif';
USmap_tif = double(imread(dir_tif));
USmap_tif(USmap_tif==0)=nan;
code_fips = unique(USmap_tif(~isnan(USmap_tif)));
% % boundary set
map_boundary = [-130, -60, 20, 50]; % [lon_min, lon_max, lat_min, lat_max]

%% main calculation
for iv = 1:length(name_mod)
    % %
    crop_mod = name_mod{iv};

    % % load obs yield anomaly
    dir_org_fips  = [f_result  '/anomaly_obs/' name_obs{iv} '_climbin_yield_anomaly_linear.csv'];
    data_org_fips = readtable(dir_org_fips);

    % % load mod result
    for icc = 1:2
        climate  = name_climate{icc};

        scenario_org = 'org_noirr';
        dir_org    = [f_result '/obs_mod_csv/yield_org/obs_mod_' crop_mod '_noirr_' climate '.csv'];
        data_org   = readtable(dir_org);

        dir_best   = [f_result '/obs_mod_csv/yield_adj/obs_mod_' crop_mod '_noirr_' climate '.csv'];
        data_best  = readtable(dir_best);

        if icc == 1
            mergedTable_org  = data_org;
            mergedTable_best = data_best;
        else
            mergedTable_org  = outerjoin(mergedTable_org, data_org, 'Keys', {'time', 'lat','lon','Area','Prec_sigma_bin','obs'}, 'MergeKeys', true);
            mergedTable_best = outerjoin(mergedTable_best, data_best, 'Keys', {'time', 'lat','lon','Area','Prec_sigma_bin','obs'}, 'MergeKeys', true);
        end
    end
    data_org  = merge_climate(data_org,mergedTable_org,'org');
    data_best = merge_climate(data_best,mergedTable_best,'best');

    %% response curve
    % % calculate average value at each Pbin
    plotdata_group_org.(crop_mod) = cal_Pbin_mean(data_org);
    plotdata_group_best.(crop_mod) = cal_Pbin_mean(data_best);

end


%% FIGURE
flag_curve = 1;
if flag_curve
    Nh=1; Nw=3;
    gap = [0.07 0.05];
    marg_h = [0.2 0.2];
    marg_w = [0.08 0.08];
    [ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w);
    set(gcf,'Position',[1000 500 680 260])

    % % colorbar data set
    cdata_O = flip(colormap(brewermap([],'Oranges')));
    cdata_G = colormap(brewermap([],'Greens'));
    cdata_B = flip(colormap(brewermap([],'Blues')));
    cdata = [cdata_B;cdata_G];
    colormap(cdata)

    color_dry = [0.8 0 0];
    color_wet = [0 0 0.8];

    color_org  = cdata_O(120,:);
    color_best = cdata_B(80,:);

    % % format set
    AB1 = {'a' 'b','c'};
    AB2 = {'a' 'b','c'};
    AB3 = {'d' 'e','f'};
    AB4 = {'g' 'h','i'};

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    %% FIGURE - response curve
    for iv = 1:length(name_mod)
        % % data for plot
        crop_mod = name_mod{iv};
        climate  = name_climate{1};

        group_org = plotdata_group_org.(crop_mod);
        group_best= plotdata_group_best.(crop_mod);

        %% plot
        axes(ha(iv))  
        plot([-3.2 4.2],[0 0],'k--');hold on;
        % % obs data
        xx = group_org.Prec_sigma_bin;
        yy = group_org.mean_obs;
        xx_add = [xx;xx(end)+0.5];
        yy_add = [yy;yy(end)];
        ll_obs = stairs(xx_add, yy_add, 'k', 'LineWidth', 1.5);
        hold on;
        % % original model results
        ll_org = plot_reponse_curve(group_org,color_org);
        % % revised model results
        ll_best = plot_reponse_curve(group_best,color_best);

        % % format
        hold off;
        xlim([-3.2 4.2]); xticks(-3:0.5:3.5);    xtickformat('%.1f');
        ylim([-80 80]); yticks(-60:20:60);
        set(gca,'FontSize',9)
        text(0.03,0.05,'Extreme dry',Units='normalized',Color=color_dry,FontSize=8)
        text(0.70,0.05,'Extreme wet',Units='normalized',Color=color_wet,FontSize=8)
        legend([ll_obs ll_org ll_best],{'Obs','AgMIP','AgMIP\_adj'},'Box','off','Location','northwest')
        xlabel('Precipitation anomaly ({\it\sigma})','FontSize',10)
        ylabel('Yield change (%)','FontSize',10)
        title(name_title{iv},FontSize=12)
        text(-0.04,1.05,AB1{iv},Units='normalized',FontWeight='bold',FontSize=12)
    end

    savefig_name = './figure_save/yield_response_us';
    userInput = input(['Do you want to save figure to ? (y/n): ' savefig_name '  '], 's');
    if strcmpi(userInput, 'y')
        disp('Operation will be executed.');
        % save figure
        exportgraphics(gcf,[savefig_name,'.pdf'],'ContentType','vector')
        exportgraphics(gcf,[savefig_name,'.jpg'],'Resolution',300)
    else
        disp('Operation canceled by user.');
    end

end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

function mergedTable = merge_climate(data,mergedTable,endstr)
name_model = data.Properties.VariableNames(7:end);
for i = 1:length(name_model)
    mergedTable{:,name_model{i}} = mean([mergedTable{:,[name_model{i},'_data_',endstr]}, ...
        mergedTable{:,[name_model{i},'_mergedTable_',endstr]}], 2, 'omitnan');
    mergedTable = removevars(mergedTable, {[name_model{i},'_data_',endstr], [name_model{i},'_mergedTable_',endstr]});
end
end

function group_org = cal_Pbin_mean(data_org)
name_model = data_org.Properties.VariableNames(7:end);

group_org = groupsummary(data_org, 'Prec_sigma_bin', 'mean', 'obs');
weightedMean = @(values, weights) mean(values .* weights,'omitnan') / mean(weights,'omitnan') * 100;
group_org.mean_obs = splitapply(weightedMean, data_org.obs, data_org.Area, findgroups(data_org.Prec_sigma_bin));

for im = 1:length(name_model)
    eval(['group_org.mean_' name_model{im} ' = splitapply(weightedMean, data_org.' name_model{im} ', data_org.Area, findgroups(data_org.Prec_sigma_bin));'])
end
group_org.model_ensemble = median(group_org{:,4:13},2);

group_org.Prec_sigma_bin = (-3:0.5:3.5)';
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function group_org = group_mean(data_org,gvar,datavar)
group_org = groupsummary(data_org, gvar, 'mean', datavar);
weightedMean = @(values, weights) mean(values .* weights,'omitnan') / mean(weights,'omitnan') * 100;
group_org.gmean = splitapply(weightedMean, data_org(:,datavar), data_org.Area, findgroups(data_org(:,gvar)));
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function [data_grid,lon_x,lat_y] = csv_to_grid_obs(group_org_fips_wet,USmap_tif,map_boundary)
grid = 0.05;
lon_range = (map_boundary(1) + grid/2):grid:(map_boundary(2) - grid/2);
lat_range = (map_boundary(4) - grid/2):-grid:(map_boundary(3) + grid/2);

data_grid = NaN(length(lat_range), length(lon_range));
for i = 1:height(group_org_fips_wet)
    idx = USmap_tif == group_org_fips_wet.FIPS(i);
    if ~isempty(idx)
        data_grid(idx) = group_org_fips_wet.gmean(i);
    end
end
[lon_x, lat_y] = meshgrid(lon_range, lat_range);

end

function [data_grid,lon_x,lat_y] = csv_to_grid(group_best_wet,map_boundary)
grid = 0.5;
lon_range = (map_boundary(1) + grid/2):grid:(map_boundary(2) - grid/2);
lat_range = (map_boundary(4) - grid/2):-grid:(map_boundary(3) + grid/2);
data_grid = NaN(length(lat_range), length(lon_range));
for i = 1:height(group_best_wet)
    lon_idx = lon_range == group_best_wet.lon(i);
    lat_idx = lat_range == group_best_wet.lat(i);
    data_grid(lat_idx, lon_idx) = group_best_wet.gmean(i);
end
[lon_x, lat_y] = meshgrid(lon_range, lat_range);
end

function [data_grid_best,lon_x_best,lat_y_best] = cal_spatial_aggregate(data_best,pBin_thres,map_boundary)
data_best.model_ensemble = median(data_best{:,7:16},2,'omitnan');  %%%%%%%%% model name index 7:16
data_best_wet = data_best(data_best.Prec_sigma_bin > pBin_thres,:);
gvar = {'lat','lon'}; datavar = 'model_ensemble';
group_best_wet = group_mean(data_best_wet,gvar,datavar);
[data_grid_best,lon_x_best,lat_y_best] = csv_to_grid(group_best_wet,map_boundary);
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function ll_mod = plot_reponse_curve(group_org,color_set)
xx = group_org.Prec_sigma_bin;
yy = group_org.model_ensemble;
y_min = prctile(group_org{:,4:13}, 25,2);
y_max = prctile(group_org{:,4:13}, 75,2);

xx_add = [xx;xx(end)+0.5];
yy_add = [yy;yy(end)];

ll_mod = stairs(xx_add, yy_add, 'color',color_set, 'LineWidth', 1.5);
hold on;
for i = 1:numel(xx_add)-1
    f_xx = [xx_add(i) xx_add(i+1)];
    f_y_min = [y_min(i) y_min(i)];
    f_y_max = [y_max(i) y_max(i)];
    fill([f_xx, fliplr(f_xx)], [f_y_min, fliplr(f_y_max)], color_set,'FaceAlpha', 0.2, 'EdgeAlpha', 0);
    hold on
end
end

function set_colorbar(minvalue,maxvalue)
ch = colorbar('southoutside');
clim([minvalue maxvalue]);
ch.Ticks = [-60:20:60];
ch.FontSize = 8;
ch.Label.String = 'Yield change (%)';
ch.Label.FontSize = 10;
pos_ch = ch.Position;
a = 0.5;
pos_ch(1) = pos_ch(1) + pos_ch(3)*(1-a)/2;
pos_ch(2) = pos_ch(2) + 0.015;
pos_ch(3) = pos_ch(3) * a;
pos_ch(4) = pos_ch(4) * 0.3;
ch.Position = pos_ch;
end

function plot_us_pattern(USmap_states,lat_y_obs,lon_x_obs,data_grid_obs,map_boundary,maplabel)
ax = worldmap([map_boundary(3) map_boundary(4)], [map_boundary(1) map_boundary(2)]); 
framem off;    gridm off;   mlabel off; plabel off; 

% % plot data
pcolorm(lat_y_obs,lon_x_obs,data_grid_obs);
geoshow(ax,[USmap_states.Y],[USmap_states.X], 'DisplayType', 'polygon', 'FaceColor', 'none', 'EdgeColor', 'k');

minvalue = -80;%min(values);
maxvalue = 80;%max(values);
clim([minvalue maxvalue]);
% format
axis off;
text(0.1,0.9,maplabel,Units='normalized',FontWeight='bold',FontSize=12)
end

