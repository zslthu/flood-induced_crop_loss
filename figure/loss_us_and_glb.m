clc;clear;
warning('off', 'all');

% % % % model information
name_crop_mod = {'maize','soy','wheat'};
name_model    = {'cgms-wofost','lpj-guess','clm-crop','lpjml','epic-iiasa','gepic','orchidee-crop','pdssat','papsim','pegasus'};
name_climate  = {'wfdei.gpcc','agmerra'};
title_name    = {'maize','soybean','wheat'};

yr_start = 1991;
yr_end   = 2009;
period   = yr_start:yr_end;

% % % % read report data
dir_report  = '../../data/natural-disasters-M49.csv';
data_report = readtable(dir_report);
idx_f       = find(contains(data_report.Properties.VariableNames,'Flood'));
data_report = data_report(:,[1,2,3,idx_f]);
data_report.TotalEconomicDamagesFromFloods = data_report.TotalEconomicDamagesFromFloods; % unit: (‘000 US$)
idx_year    = data_report.Year>=yr_start & data_report.Year<=yr_end;
data_report = data_report(idx_year,:);

% % % % read country code and boundary
dir_country  = '../../data/crop_price/M49.nc';
data_country = flip(ncread(dir_country,'Band1')');
code_country = unique(data_country(~isnan(data_country)));
num_country  = length(code_country);
M49_line = shaperead('../../data/M49_line/m49_line.shp');


%% US
f_result     = '../../output/obs_mod_csv/indemnity';
loss_type    = 'usd';
[plotdata_group_best, plotdata_group_best_year] = cal_us_indemnity(f_result, loss_type);

%% global
f_result    = '../../output/loss_glb';
[loss_compare,loss_compare_crop, perform_ci, data_loss, data_loss_all] = cal_global_indemnity(f_result, loss_type);

% % subplot set
Nh=2; Nw=3; % subplot
gap = [0.09 0.08];
marg_h = [0.11 0.09];
marg_w = [0.08 0.08];
[ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w);
set(gcf,'Position',[1000 500 650 440])

axes(ha(6));axis off;

% % color set
cdata = colormap(brewermap([],'YlGnBu'));%colormap("parula");...
cdata([230;end],:) = [];
cdata_crop = cdata([200 120 60],:);
cdata_obs = [0.8 0.4 0.1];

% % format set
marker_crop = {'p','^','o'};
marker_size = [6 4 3.5];
AB1 = {'a','b','c'};
AB2 = {'d','e','f'};
cns_pos = [0.68 0.68 0.58];

%% [FIGURE] - US
flag_us = 1;
if flag_us
    for iv = 1:3
        crop_mod = name_crop_mod{iv};

        group_best = plotdata_group_best.(crop_mod);
        group_best_year = plotdata_group_best_year.(crop_mod);
 
        %% plot temporal comparsion
        period = group_best_year.time;
        xx = group_best_year.sum_indemnity;
        yy = group_best_year.model_ensemble;

        axes(ha(iv))
        pos_change = cell2mat(pos(iv));
        pos_change(4) = pos_change(4) * 0.9;
        set(gca,'Position',pos_change)

        ax1 = gca;
        % % mod results
        yyaxis left;
        hb=bar(period,yy,'EdgeColor','none','FaceColor',cdata_crop(1,:),'BarWidth',0.6,'FaceAlpha',0.4);
        hold on;
        set(gca,'ycolor','k')
        set(gca,'yscale','log')
        ylim_min = 1e6*3;
        ylim_max = 1e13*2;
        set(gca,'ylim',[ylim_min ylim_max],'YTick',[1e7 1e9 1e11 1e13])
        set(gca,'xlim',[1990 2010],'xticklabel',[1990:5:2010])
        set(gca,'fontsize',8)
        if iv == 1; ylabel('Estimated loss in US (USD)','fontsize',10); end

        % % reported indmenity
        yyaxis right;
        ax1.YAxis(2).Color = cdata_obs;
        hp=plot(period,xx,'Color',cdata_obs,'LineStyle','--','Marker',marker_crop{iv},'MarkerSize',marker_size(iv),'LineWidth',0.8);
        hold on;
        set(gca,'ycolor',cdata_obs)
        set(gca,'yscale','log')
        ylim_min = 1e6*2;
        ylim_max = 1e12*0.9;
        set(gca,'ylim',[ylim_min ylim_max],'YTick',[1e7 1e8 1e9 1e10 1e11])
        set(gca,'xlim',[1990 2010])
        if iv == 3; ylabel({'Reported indemnity in US (USD)'},'fontsize',10); end

        % % format
        text(-0.05,1.06,AB1{iv},'units','normalized','fontsize',12,'FontWeight','bold')

        idx_nonan = ~isnan(xx+yy);
        xx = xx(idx_nonan);
        yy = yy(idx_nonan);
        top_num = 10;
        perform_idx = cal_cns(xx,yy,top_num);
        text(0.7,0.9,['{\itCI} = ' num2str(perform_idx,'%0.2f')],'units','normalized','fontsize',10)
        
        title(title_name{iv},'fontsize',11)

        %% 1:1 compare - spatial pattern
        flag_insert = 1;
        if flag_insert
            pos_bar=pos_change;
            wx = pos_bar(3)*0.32; wy = pos_bar(4)*0.24;
            lx = pos_bar(1)+0.035;
            ly = pos_bar(2)+0.23;
            axes('Position',[lx ly wx wy])

            xx = group_best.sum_indemnity;
            yy = group_best.model_ensemble;

            delete_idx = find((yy==0) | (xx==0));
            xx(delete_idx)=[];
            yy(delete_idx)=[];

            xx_save = xx;
            xx = log10(yy);
            yy = log10(xx_save);

            plot(xx,yy,marker_crop{iv},'Color',cdata_crop(1,:),'MarkerSize',marker_size(iv)*0.6,'LineWidth',0.5,...
                'MarkerFaceColor',cdata_crop(1,:),'MarkerEdgeColor','none');
            [p,S] = polyfit(xx,yy,1);
            [yy_fit,delta] = polyval(p,xx,S);
            hold on;
            plot(xx,yy_fit,'k-','LineWidth',0.5)
            hold on; plot([3 11],[3 11],'k-')

            % calculate R2 p-value and RMSE
            [R,p_value] = corr(10.^xx,10.^yy); %p_value
            R2 = R^2;
            if p_value<0.01
                p_level = '**';
            elseif p_value<0.05
                p_level = '*';
            else
                p_level = '';
            end

            text(0.07,0.92,{['{\itR}^2 = ' num2str(R2,'%0.2f'),'^{',p_level,'}']},...
                'Units','normalized','fontsize',8)
            % % format
            box off;
            set(gca,'fontsize',5)
            set(gca,'ytick',[5 7 9],'YTickLabel',{'10^5' '10^7' '10^9' '10^{10}'})
            set(gca,'xtick',[5 7 9],'XTickLabel',{'10^5' '10^7' '10^9' '10^{10}'})
            xlabel('Estimated loss','fontsize',7)
            text(0.05,-0.6,'in each state','Units','normalized','FontSize',7)
            ylabel('Indemnity','fontsize',7)
        end
    end
end

%% [FIGURE] - GLB
% figure;
fig_global = 1;
if fig_global
    axes(ha(4))

    for icc = 1:3
        loss_temp = loss_compare_crop{icc};
        yusd_crop(:,icc) = loss_temp.loss_mod/1e9;
    end
    xx = 1991:2009;
    y1 = loss_compare.loss_obs/1e9;
    y3 = loss_compare.loss_mod/1e9;
    
    ax1 = gca;
    % % left
    yyaxis left;
    ax1.YAxis(1).Color = [0 0 0];
    b = bar(xx,yusd_crop,'stacked','FaceColor','flat','EdgeColor','none','BarWidth',0.7);
    for k = 1:3
        b(k).CData = cdata_crop(k,:);
    end
    % % format
    set(gca,'XTicklabel',[1990:2:2010]);
    set(gca,'fontsize',8);
    text(-0.15,0.05,'Global estimated loss (B USD)','Units','normalized','Rotation',90,'FontSize',10);
    ylim([0,max(y3)*1.3])
    
    % % right
    yyaxis right;
    ax1.YAxis(2).Color = cdata_obs;
    plot(xx,y1, 'Marker','o','MarkerSize',4,'LineWidth',1,'LineStyle','--','Color',cdata_obs); % area
    % % format
    ytickformat = '%.1f';
    ylim([0 50])
    set(gca,'XTicklabel',[1990:5:2010]);    
    set(gca,'fontsize',8)
    legend(title_name,'Box','off','FontSize',8)
    ylabel('Global loss from EM-DAT (B USD)','FontSize',10)

    % % add text
    [R,p_value] = corr(y3,y1); %p_value
    R2 = R^2;
    if p_value<0.01
        p_label = ['< 0.01'];
    else
        p_label = ['= ' num2str(floor(p_value*100)/100,'%0.2f')];
    end
    text(0.05,0.9,['{\itCI} = ' num2str(perform_ci,'%0.2f')],'Units','normalized','FontSize',10)
    text(-0.05,1.05,'d','Units','normalized','FontSize',13,'FontWeight','bold')
    
end

map_country= 1;
if map_country
    %% load and process results
    % % % % [1]report data
    data_report_country = rmmissing(data_report(:,{'CountryName','M49','Year','TotalEconomicDamagesFromFloods'}));
    [uniqueAreas, ~, idx] = unique(data_report_country.M49);
    avg_data_obs = table(uniqueAreas, 'VariableNames', {'M49'});
    for i= 1:length(uniqueAreas)
        avg_data_obs.CountryName(i) = data_report_country.CountryName(find(data_report_country.M49==uniqueAreas(i),1));
        avg_data_obs.Avg_loss(i)    = mean(data_report_country.TotalEconomicDamagesFromFloods(data_report_country.M49==uniqueAreas(i)));
    end
    avg_data_obs = sortrows(avg_data_obs,'Avg_loss','descend');
    avg_data_obs(1,:) = [];

    % % % % [2] mod data
    avg_data_mod = data_loss(:,{'Area','Area_Code_M49_'});
    avg_data_mod = sortrows(avg_data_mod,"Area_Code_M49_",'ascend');
    for ic = 1:3
        v_crop = name_crop_mod{ic};
        data_temp  = data_loss_all(strcmp(data_loss_all.Item,v_crop),:);
        data_temp  = sortrows(data_temp,"Area_Code_M49_",'ascend');
        idx_y        = find(contains(data_temp.Properties.VariableNames,'Y')); idx_y = idx_y(1); 
        avg_data_mod{:,['Avg_loss_' v_crop]} = mean(data_temp{:,idx_y:end},2);
    end
    avg_data_mod.Avg_loss = sum(avg_data_mod{:,{'Avg_loss_maize','Avg_loss_soy','Avg_loss_wheat'}},2);
    avg_data_mod = sortrows(avg_data_mod,'Avg_loss','descend');
    avg_data_mod(1,:) = [];

    code_union   = data_loss.Area_Code_M49_;
    data_glbloss = data_country;
    for i = 1:length(code_union)
        area_code = code_union(i);
        loss_temp = avg_data_mod{avg_data_mod.Area_Code_M49_==area_code,'Avg_loss'};
        if ~isempty(loss_temp)
            data_glbloss(data_country==area_code) = log10(loss_temp);
        end
    end

    perform  = ones(length(code_union),1) * nan;
    for i = 1:length(code_union)
        loss_mod      = table();
        loss_mod.Year = (yr_start:yr_end)';
        area_code = code_union(i);
        [~, perform(i,:)] = cal_perform(data_report,data_loss,loss_mod,area_code);
    end

    %% [FIGURE]
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    pi=5;
    axes(ha(pi))
    pos_change = cell2mat(pos(pi));
    pos_change(3) = pos_change(3) * 2.5;
    % pos_change(4) = pos_change(4) * 1.2;
    set(gca,'position',pos_change)

    % % plot data
    plot_glb_country(M49_line,data_glbloss,cdata,1)
    text(-0.03,1.05,'e','Units','normalized','FontSize',13,'FontWeight','bold')

    %% insert: histogram
    pos_bar = get(gca,'Position');
    wx = pos_bar(3)*0.14; wy = pos_bar(4)*0.3;
    lx = pos_bar(1)+0.042;
    ly = pos_bar(2)+0.055;
    axes('Position',[lx ly wx wy])

    histogram(perform,'FaceColor', cdata(160,:))
    box off;
    set(gca,'xlim',[0 1],'xtick',[0.2:0.2:0.8])
    ylabel('Number of country','fontsize',10)
    xlabel('\itCI','fontsize',10)
    set(gca,'fontsize',7)
end


% % % % % % % % % % % % % % % % % % % % % % % % % %
savefig_name = ['./figure_save/loss_us_and_glb'];
userInput = input(['Do you want to save figure to ? (y/n): ' savefig_name '  '], 's');
if strcmpi(userInput, 'y')
    disp('Operation will be executed.');
    % save figure
    exportgraphics(gcf,[savefig_name,'.pdf'],'ContentType','vector')
    exportgraphics(gcf,[savefig_name,'.jpg'],'Resolution',300)
else
    disp('Operation canceled by user.');
end


%==================function
function perform_idx = cal_cns(xx,yy,len)
[loss_mod_temp,loss_mod_sort_idx] = sort(yy,'descend');
loss_mod_sort_idx = loss_mod_sort_idx(1:len);
[loss_obs_temp,loss_obs_sort_idx] = sort(xx,'descend');
loss_obs_sort_idx = loss_obs_sort_idx(1:len);

perform_idx = length(intersect(loss_obs_sort_idx,loss_mod_sort_idx))/length(loss_obs_sort_idx);
end

function plot_glb_country(M49_line,data_glbloss,cdata,c_flag)

plot(([M49_line.X]+180)*20,([M49_line.Y]+90)*20,'-','Color',[0.5 0.5 0.5],'LineWidth',0.3)
hold on;
data_plot = flip(data_glbloss);
hl = imagesc(data_plot);
set(hl,'AlphaData',~isnan(data_plot))
hold on;
plot(([M49_line.X]+180)*20,([M49_line.Y]+90)*20,'-','Color',[0.5 0.5 0.5],'LineWidth',0.3)

% % format
set(gca,'ylim',[601 3600],'xlim',[1 7200])
set(gca,'xtick',[],'ytick',[])
colormap(cdata)
caxis([4 11]);

if c_flag
% % colorbar
ch=colorbar;
ch.FontSize = 8;
ch.Ticks = 5:10;
ch.TickLabels = {'10^5' '10^6' '10^7' '10^{8}' '10^{9}' '10^{10}' };
ch.Label.String = 'Global estimated loss (B USD)';
ch.Label.FontSize = 10;
end

end

function [loss_compare, perform_idx] = cal_perform(data_report,data_loss,loss_mod, area_code)

idx_y    = find(contains(data_loss.Properties.VariableNames,'Y')); idx_y = idx_y(1);
loss_mod.loss_mod = data_loss{data_loss.Area_Code_M49_==area_code,idx_y:end}';

idx_world = data_report.M49 == area_code;
loss_obs = data_report(idx_world,{'Year','TotalEconomicDamagesFromFloods'});

loss_compare = outerjoin(loss_mod,loss_obs,'Keys','Year','MergeKeys', true);
loss_compare = rmmissing(loss_compare); 
loss_compare.Properties.VariableNames = {'Year','loss_mod','loss_obs'};

if sum(loss_mod.loss_mod)~=0 && size(loss_compare,1) >= 8
    [~,loss_mod_sort_idx] = sort(loss_compare.loss_mod,'descend');
    loss_mod_sort_idx = loss_mod_sort_idx(1:round(end/2));

    [~,loss_obs_sort_idx] = sort(loss_compare.loss_obs,'descend');
    loss_obs_sort_idx = loss_obs_sort_idx(1:round(end/2));

    perform_idx = length(intersect(loss_obs_sort_idx,loss_mod_sort_idx))/length(loss_obs_sort_idx);
else
    perform_idx = nan;
end
end