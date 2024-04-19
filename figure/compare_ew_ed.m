clc;clear

% % % % model information
name_crop     = {'maize','soy','wheat'};
title_crop    = {'maize','soybean','wheat'};
name_model    = {'crover','epic-iiasa', 'ldndc', 'lpj-guess', 'pdssat', 'pepic', 'promet', 'simplace-lintul5'}; 
name_climate  = {'gfdl-esm4','ukesm1-0-ll','mri-esm2-0','mpi-esm1-2-hr','ipsl-cm6a-lr'};

name_ssp = {'ssp126' 'ssp585'};
name_dw  = {'dry','wet'};

% % % % file
dir_result     = '../../output/loss_glb_ssp/ssp_adj';
dir_result_org = '../../output/loss_glb_ssp/ssp_org';

% % subplot set
Nh=1; Nw=3; 
gap = [0.09 0];
marg_h = [0.15 0.1];
marg_w = [0.08 0.4];
[ha, pos] = tight_subplot(Nh, Nw, gap, marg_h, marg_w);
set(gcf,'Position',[1000 500 650 260])

color_set = [0.1 0.5 1;
    0.9 0.1 0;
    0 0 0;
    0.1 0.5 1;
    0.9 0.1 0];
color_gray = [0.8 0.8 0.8];
alpha_set  = [0.25 0.25 0 0.1 0.1];

AB  = {'a' 'b' 'c'};

% % % % country info
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

        %% adjusted vs original
        for id = 1:2
            dir_loss = [dir_result '/' ssp '/compare_ew_ed/' name_dw{id} '_loss_usd_' crop '_ensemble_' ssp '.csv'];
            data_loss = readtable(dir_loss);
            data_bar(:,id+3) = data_loss{1,4:end};
        end

        for id = 1:2
            add = name_dw{id};
            dir_loss = [dir_result_org '/' ssp '/compare_ew_ed/' name_dw{id} '_loss_usd_' crop '_ensemble_' ssp '.csv'];
            data_loss = readtable(dir_loss);
            data_bar(:,id) = data_loss{1,4:end};
        end
        data_bar(data_bar==0)=nan;


        %% plot
        axes(ha(iv))

        data_bar = data_bar/10^9;
        for ii = 1:5
            y = data_bar(:,ii);
            x = (rand(1, length(y))-0.5)*0.4+ii; 
            jitterAmount = 0.02; 
            x_jittered = x + jitterAmount * randn(size(x));
            scatter(x_jittered,y,7,color_gray); hold on;  
        end

        hb = boxplot(data_bar,'Symbol','','Colors','k');
        h = findobj(gca,'Tag','Box');

        for i = 1:length(h)
            patch(get(h(i),'XData'),get(h(i),'YData'),color_set(i,:),'FaceAlpha',alpha_set(i),...
                'Edgecolor',color_set(i,:),'EdgeAlpha',alpha_set(i)*2,'LineWidth',1);
        end
        medians = findobj(gca, 'tag', 'Median');
        for j = 1:length(medians)
            set(medians(j), 'Color', 'k', 'LineWidth', 1.25); 
        end

        %%% format       
        hold on; plot([3 3],[0 2*10^9],'k--','LineWidth',0.5)
        ylim([0 3]) 

        %%% add text
        text(0.1,0.9,'original','Units','normalized','FontSize',10)
        text(0.6,0.9,'adjusted','Units','normalized','FontSize',10)
        title(title_crop{iv})
        if iv == 1
            text(-0.17,1.07,AB{iv},'Units','normalized','FontSize',13,'fontweight','bold')
            text(-0.28,0.12,'Projected annual loss (B USD)','Units','normalized','FontSize',10,'Rotation',90)
        end
        if iv ~=1
            set(gca,'ytick',[])
        end

        ytickformat('%.1f');
        set(gca,'xticklabel',{'ED','EW','','ED','EW'},'xticklabelRotation',90)

        dir_result_txt=dir_result;
        for li=1:2
            if li==1
                dir_result = dir_result_org;
            else
                dir_result = dir_result_txt;
            end
            %% global ed loss / ew loss each country
            dir_loss = [dir_result '/' ssp '/compare_ew_ed/wet_loss_usd_' crop '_ensemble_' ssp '.csv'];
            data_loss_wet = readtable(dir_loss);

            dir_loss = [dir_result '/' ssp '/compare_ew_ed/dry_loss_usd_' crop '_ensemble_' ssp '.csv'];
            data_loss_dry = readtable(dir_loss);

            data_country_trend = zeros(size(data_loss_wet,1),1)*nan;
            for icy = 1:size(data_loss_wet,1)
                data_country_trend(icy,1) = mean(data_loss_wet{icy,4:end})./mean(data_loss_dry{icy,4:end});
            end
            data_trend = data_loss_wet(:,1:3);
            data_trend = [data_trend array2table(data_country_trend,'VariableNames',{'ew_ed'})];

            %% plot
            comName = 'M49';
            data_trend_info = innerjoin(data_trend, country_info, 'Keys', comName);

            % continent_ID = unique(data_trend_info.Continent_ID);
            continent_ID = unique(data_trend_info(:,{'Continent_ID','Continent_EN_Name'}));
            xlabelname = ['Global'; continent_ID.Continent_EN_Name];

            % % plot data
            y_glb = data_trend_info.ew_ed;
            ratio(iv+(li-1)*3,1) = sum(y_glb>1)/sum(y_glb>=0)*100;
            for jj = 1:height(continent_ID)
                idx = data_trend_info.Continent_ID == continent_ID.Continent_ID(jj);
                data_trend_info_c = rmmissing(data_trend_info(idx,:));
                % plot
                y = data_trend_info_c.ew_ed;
                % positive & negative ratio
                ratio(iv+(li-1)*3,jj+1) = sum(y>1)/length(y)*100;
            end
 
        end
    end
    pos_hm = cell2mat(pos(3));
    pos_hm(1) = pos_hm(1)+0.29;
    pos_hm(3) = pos_hm(3)*0.8;
    axes('Position',pos_hm)

    xvalues = title_crop;
    yvalues = xlabelname;
    heatmapObject = heatmap(ratio');
    heatmapObject.XDisplayLabels = [title_crop,title_crop];%repmat({''}, size(xvalues));
    heatmapObject.YDisplayLabels = xlabelname;%repmat({''}, size(xvalues));
    cdata = colormap(brewermap([],'YlGnBu'));%'Spectral'));
    colormap(cdata)
    % clim([0 100])
    caxis([0 100])
    
    %%% format
    textAxes = axes('Position', pos_hm, 'Color', 'none');
    axis(textAxes, 'off'); 
    text(1.4,0,{'Countries with more losses due to EW (%)'},'Units','normalized','Rotation',90)
    text(0,1.035,{'  original    adjusted'},'Units','normalized')
    text(-0.12,1.05,'b','Units','normalized','FontSize',13,'fontweight','bold')
end



savefig_name = ['./figure_save/compare_ew_ed_' ssp];
userInput = input(['Do you want to save figure to ? (y/n): ' savefig_name '  '], 's');
if strcmpi(userInput, 'y')
    disp('Operation will be executed.');
    % save figure
    exportgraphics(gcf,[savefig_name,'.pdf'],'ContentType','vector')
    exportgraphics(gcf,[savefig_name,'.jpg'],'Resolution',300)
    % print(gcf, [savefig_name,'.jpg'], '-djpeg', '-r0');
else
    disp('Operation canceled by user.');
end

