clc;clear

% % % % model information
name_crop    = {'maize','soy','wheat'};
name_model   = {'crover','epic-iiasa', 'ldndc', 'lpj-guess', 'lpjml', 'pdssat', 'pepic', 'promet', 'simplace-lintul5'};
name_climate = {'gfdl-esm4','ukesm1-0-ll','mri-esm2-0','mpi-esm1-2-hr','ipsl-cm6a-lr'};
name_ssp     = {'ssp126' 'ssp585'};

dir_result = '../../output/loss_glb_ssp/ssp_adj';

% % plot set
Nh=1; Nw=3; % subplot
gap = [0.05 0.05];
marg_h = [0.2 0.08];
marg_w = [0.08 0.08];
[ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w);
set(gcf,'Position',[1000 500 730 350])

cdata = colormap(brewermap([],'YlGnBu'));
cdata_crop = cdata([200 130 90],:);

AB = {'a' 'b' 'c'};

% % country info
country_info = readtable('../../data/country_code/country_code.csv');
names = country_info.Properties.VariableNames;
names{1} = 'M49';
country_info.Properties.VariableNames = names;
country_info.Continent_EN_Name(strcmp(country_info.Continent_EN_Name,'Asia')) = {'Asia & Oceania'};
country_info.Continent_EN_Name(strcmp(country_info.Continent_EN_Name,'Oceania')) = {'Asia & Oceania'};
country_info.Continent_EN_Name(strcmp(country_info.Continent_EN_Name,'North America')) = {'N. America'};
country_info.Continent_EN_Name(strcmp(country_info.Continent_EN_Name,'South America')) = {'S. America'};
country_info.Continent_ID(country_info.Continent_ID == 4) = 1;
country_info.Continent_ID(country_info.Continent_ID == 3) = 5;
country_info.Continent_ID(country_info.Continent_ID == 6) = 3;
country_info.Continent_ID(country_info.Continent_ID == 7) = 4;

%% main
for is = 2
    ssp = name_ssp{is};
    for iv = 1:3
        crop = name_crop{iv};
        dir_loss = [dir_result '/' ssp '/loss_usd/loss_usd_' crop '_ensemble_' ssp '.csv'];
        data_loss = readtable(dir_loss);

        %% global loss
        yr_start = 2015;
        yr_end   = 2100;
        step     = 5;
        p_start  = 2021;
        period   = p_start:step:yr_end;
        icy = 1;
        temp = data_loss{icy,4:end};
        for pi = 1:(yr_end - p_start+1)/step
            pii = (pi-1)*step+(p_start-yr_start+1);
            data_global_year(pi,iv) = mean(temp(pii:pii+step-1));
        end
        mean(data_global_year)

        %% global model range
        imic = 0; data_global_tmp = [];
        for im = 1:length(name_model)
            for ic = 1:length(name_climate)
                imic = imic +1;
                model_climate = [name_model{im} '_' name_climate{ic}];
                dir_loss_model = [dir_result '/' ssp '/loss_usd/loss_usd_' crop '_' model_climate '_' ssp '.csv'];;
                data_loss_model = readtable(dir_loss_model);

                icy = 1;
                temp = data_loss_model{icy,4:end};
                for pi = 1:(yr_end - p_start+1)/step
                    pii = (pi-1)*step+(p_start-yr_start+1);
                    data_global_tmp(pi,imic) = mean(temp(pii:pii+step-1));
                end
            end
        end
        data_model_std(:,iv) = std(data_global_tmp,[],2)/2;

        %% plot temproal
        % subplot(2,3,iv)
        % axes(ha(iv))
        pos_c = cell2mat(pos(iv));
        b = 0.25;
        pos_c(2) = pos_c(2)+pos_c(4)*(1-b*1.35);
        pos_c(4) = pos_c(4)*b;
        a = 0.6;
        pos_c(1) = pos_c(1)+0.04;
        pos_c(3) = pos_c(3)*a;
        axes('Position',pos_c)

        tloc = [0.95,0.85,0.75];
        alp  = [0.15,0.15,0.25];
        AB = {'b','c','d'};

        data_crop = data_global_year(:,iv)/10^9;
        data_d = data_crop-data_model_std(:,iv)/10^9;
        data_u = data_crop+data_model_std(:,iv)/10^9;

        plot(period,data_crop,'-','LineWidth',1.5,'color',cdata_crop(iv,:));hold on;
        hold on;
        [slope,r2,p_value]= plot_trend(data_crop,period);
        if p_value>0.01
            reg_text = ['trend = ' num2str(slope,'%.1f') '%/dec ({\itp} = ' num2str(p_value,'%.2f') ')'];
        else
            reg_text = ['trend = ' num2str(slope,'%.1f') '%/dec ({\itp} < 0.01)'];
        end
        text(0.05,0.9,reg_text,'Units','normalized','FontSize',10,'Color',cdata_crop(iv,:))

        % % add model range
        x = period;
        yd = data_d;
        yu = data_u;
        H_F = fill([x,fliplr(x)],[yd',fliplr(yu')],cdata_crop(iv,:),'FaceAlpha',alp(iv) ,'LineStyle', 'none');       
        hold on

        % % format
        xlim([2016 2100])
        ymin = min(data_d(:))*0.95;
        ymax = max(data_u(:));
        ymax = ymax+(ymax-ymin)*0.3;
        ylim([ymin ymax])
        ylim([0 9])

        set(gca,'fontsize',9,'color','none')
        title('Global loss (billion USD)','FontSize',9,'FontWeight','normal')
        box off;

        %% trend to each country
        data_country_trend = zeros(size(data_loss,1),2)*nan;
        for icy = 1:size(data_loss,1)
            temp = data_loss{icy,4:end};

            data_country_year_tmp=[];
            for pi = 1:(yr_end - p_start+1)/step
                pii = (pi-1)*step+(p_start-yr_start+1);
                data_country_year_tmp(pi,1) = median(temp(pii:pii+step-1));
            end
            if sum(isnan(data_country_year_tmp)) == 0 && sum(data_country_year_tmp) ~= 0
                [data_country_trend(icy,1),data_country_trend(icy,2)] = trendOne(data_country_year_tmp,period);        
            end
        end
        data_trend = data_loss(:,1:3);
        data_trend = [data_trend array2table(data_country_trend,'VariableNames',{'trend','trend_p'})];
        data_avg = mean(data_loss{:,4+(p_start-yr_start):end},2,'omitnan');
        data_trend = [data_trend array2table(data_avg,'VariableNames',{'average'})];

        %% data_trend + country info
        comName = 'M49';
        data_trend_info = innerjoin(data_trend, country_info, 'Keys', comName);

        continent_ID = unique(data_trend_info(:,{'Continent_ID','Continent_EN_Name'}));
        % plot
        % subplot(2,3,iv+3)
        axes(ha(iv))
        plot([0 max(continent_ID.Continent_ID)+1],[0 0],'k--');hold on
        y_boxplt = []; x_boxplt = [];
        for jj = 1:height(continent_ID)
            idx = data_trend_info.Continent_ID == continent_ID.Continent_ID(jj);
            data_trend_info_c = rmmissing(data_trend_info(idx,:));
            % plot
            y = data_trend_info_c.trend;
            x = (rand(1, length(y))-0.5)*0.4+continent_ID.Continent_ID(jj); 
            jitterAmount = 0.02; 
            x_jittered = x + jitterAmount * randn(size(x));
            scatter(x_jittered, y,8); hold on;
            y_boxplt = [y_boxplt;y];
            x_boxplt = [x_boxplt;ones(length(y),1)*jj];
            % positive & negative ratio
            ratio(iv,jj) = sum(y>0)/length(y)*100;
        end
        boxplot(y_boxplt, x_boxplt,"Colors","k","Symbol",'','Widths',0.4)
        % % format
        set(gca,'xtick',continent_ID.Continent_ID,'xticklabel',continent_ID.Continent_EN_Name)
        ylim([-20 55])
        xlim([0 max(continent_ID.Continent_ID)+1])
        if iv==1; ylabel('Trend in flood-induced loss (%/dec)'); end;
        text(-0.15,1.05,AB{iv},'Units','normalized','FontSize',14,'FontWeight','bold')
        set(gca, 'Color', 'none');
        title(name_crop{iv},'FontSize',14)

    end

    savefig_name = ['./figure_save/loss_trend_' ssp];
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


function [slope,p_value] = trendOne(sum_global,period)
[trend_global,~,~,~,stats] = regress(sum_global,[ones(length(period),1) period']);
r2      = stats(1);
p_value = stats(3);
slope   = trend_global(2)/mean(sum_global)*100 * 10;  %%% to decade
end

function [slope,r2,p_value]= plot_trend(sum_global,period)
[trend_global,~,~,~,stats] = regress(sum_global,[ones(length(period),1) period']);
plot(period,trend_global(1)+trend_global(2)*period,'k--','LineWidth',1)
r2      = stats(1);
p_value = stats(3);
slope   = trend_global(2)/mean(sum_global)*100 * 10;  %%% to decade
end