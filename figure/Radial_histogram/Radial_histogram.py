#!/home/wumej22/anaconda3/bin/python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
import matplotlib.cm as cm
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
warnings.filterwarnings('ignore')


def preprocess(v_crop):
    '''
    预处理数据, 添加Continent列
    '''
    
    country_code_fil = "./data/country_code.csv"
    country_code = pd.read_csv(country_code_fil, sep=',')
    
    crop_data_fil = f'{f_data}/{var}_{v_crop}.csv'
    print(crop_data_fil)
    crop_data = pd.read_csv(crop_data_fil, sep=',')
    crop_data = crop_data.rename(columns={'M49': 'AreaCode_M49_'})
    crop_data = crop_data.rename(columns={'Country name': 'Area'})
    crop_data.insert(3, 'Continent', None)
    
    for code in crop_data['AreaCode_M49_']:
        if code not in country_code['M49_Code'].values:
            # print (code, crop_data.loc[crop_data['AreaCode_M49_'] == code, 'Area'].values[0])
            continue
        crop_data.loc[crop_data['AreaCode_M49_'] == code, 'Continent'] = country_code.loc[country_code['M49_Code'] == code, 'Continent_EN_Name'].values[0]
    
    crop_data.loc[crop_data['AreaCode_M49_'] == 0, 'Continent'] = 'Global'
    # crop_data['Continent'] 中的 'Asia' 和 'Oceania' 改为'Asia and Oceania'
    crop_data.loc[crop_data['Continent'] == 'Asia', 'Continent'] = 'Asia & Oceania'
    crop_data.loc[crop_data['Continent'] == 'Oceania', 'Continent'] = 'Asia & Oceania'
    crop_data.loc[crop_data['Continent'] == 'North America', 'Continent'] = 'N. America'
    crop_data.loc[crop_data['Continent'] == 'South America', 'Continent'] = 'S. America'

    crop_data = crop_data.dropna(axis=0, how='any')
    crop_data = crop_data[(crop_data.T != 0).all()]
    crop_data = crop_data[~crop_data.isin([np.inf]).any(1)]

    global_loss_fil =  f"{f_data}/global_{var}_{v_crop}.csv"    
    global_loss_data = pd.DataFrame(columns=['Category', 'Value', 'StdDev'])
    global_loss = pd.read_csv(global_loss_fil, sep=',')
    # 求ssp126和ssp585的平均值和标准差
    for ssp in ['ssp126', 'ssp585']:
        global_loss_data.loc[global_loss_data.shape[0]] = [ssp, global_loss[ssp].mean(), global_loss[ssp].std()]  
    # print(global_loss_data)

    # 每个洲只保留前10的数据
    crop_data = crop_data.groupby('Continent').apply(lambda x: x.nlargest(10, ssp_f)).reset_index(drop=True)

    # 标准化：所有的数值/max_value*8
    # crop_data[ssp_f] = crop_data[ssp_f] / max_value * 8
    if var == 'loss_usd':
        crop_data[ssp_f] = np.log10(crop_data[ssp_f]) / max_value * 8
    else:
        crop_data[ssp_f] = crop_data[ssp_f] / max_value * 8

    return crop_data, global_loss_data




def Radial_histogram(v_crop, data_in, pry_cat_colname, sec_cat_colname, data_levels, **kwargs):
    '''
    
                --- 此函数用于绘制极坐标堆叠条形图 --- 
    数据分三级，一级为主要分类，二级为次要分类，三级为数据级别(堆叠分类)。
    
    必选参数：
        data_in             (DataFrame) : 包含数据的DataFrame                            
        pry_cat_colname           (str) : 主要分类的列名                             
        sec_cat_colname           (str) : 次要分类的列名                             
        data_levels              (list) : 数据级别的列名列表
        
    可选参数：                            
        primary_cats             (list) : 主要分类的列表。默认为数据中的所有唯一主要分类              
        secondary_cats           (list) : 次要分类的列表。默认为数据中的所有唯一次要分类              
        inner_circle_radius     (float) : 内圆的半径。默认为0                            
        blank_length              (int) : 每个主要分类之间的空白条形数。默认为3                 
        levels_color             (list) : 每个数据级别的颜色。默认为蓝色调色板的颜色                
        radii                    (list) : 每个数据级别的半径。默认为数据级别的最大值和总和的最大值之间的5个等距值 
        ylims                    (list) : [ymin, ymax], y轴的最小值和最大值。默认为数据级别的最大值和总和的最大值
        sort_by_Total            (bool) : 是否按总和对次要分类进行排序。默认为True               
        sort_ascending           (bool) : 是否按升序对次要分类进行排序。默认为False              
        bar_linestyle             (str) : 条形的线条样式。默认为虚线                       
        bar_linewidth           (float) : 条形的线条宽度。默认为1                          
        bar_edgecolor             (str) : 条形的边缘颜色。默认为白色                       
        bar_alpha               (float) : 条形的透明度。默认为1                           
        circle_label_fontsize     (int) : 圆圈标签的字体大小。默认为10                     
        circle_label_fontcolor    (str) : 圆圈标签的字体颜色。默认为黑色   
        circle_label_fontweight   (str) : 圆圈标签的字体粗细。默认为normal                  
        circle_linestyle          (str) : 圆圈的线条样式。默认为虚线                       
        circle_linewidth        (float) : 圆圈的线条宽度。默认为1                          
        circle_edgecolor          (str) : 圆圈的边缘颜色。默认为灰色                       
        circle_alpha            (float) : 圆圈的透明度。默认为1                           
        circle_fill              (bool) : 是否填充圆圈。默认为False                      
        bottom_circle_linestyle   (str) : 底部圆圈的线条样式。默认为实线                     
        bottom_circle_linewidth (float) : 底部圆圈的线条宽度。默认为2                        
        bottom_circle_linecolor   (str) : 底部圆圈的线条颜色。默认为黑色                     
        pry_fontsize              (int) : 主要分类标签的字体大小。默认为13                   
        pry_fontcolor             (str) : 主要分类标签的字体颜色。默认为黑色       
        pry_fontweight            (str) : 主要分类标签的字体粗细。默认为bold            
        sec_fontsize              (int) : 次要分类标签的字体大小。默认为10                   
        sec_fontcolor             (str) : 次要分类标签的字体颜色。默认为黑色
        pry_fontweight            (str) : 次要分类标签的字体粗细。默认为normal                   
        title                     (str) : 图表的标题                               
        title_fontsize            (int) : 图表标题的字体大小。默认为15                     
        title_fontcolor           (str) : 图表标题的字体颜色。默认为黑色 
        title_fontweight          (str) : 图表标题的字体粗细。默认为normal
        legend_on                (bool) : 是否显示图例。默认为True     
        legend_label_fontsize     (int) : 图例标签的字体大小。默认为10               
        legend_bbox              (list) : 图例的位置[横坐标，纵坐标]。默认为[0.5, 0.5]。[0,0]为左下角，[1,1]为右上角
        offset_pry_text         (float) : 主要分类标签的偏移量。默认为-0.5
        offset_inner            (float) : 内圆圈的偏移量。默认为-0.5
        figshow_on               (bool) : 是否显示图表。默认为True
        
        plot_subfig              (bool) : 是否绘制子图。默认为True
        subdata_in          (DataFrame) : 包含子图数据的DataFrame。默认为None
        subfig_width              (str) : 子图的宽度。默认为"25%"
        subfig_height             (str) : 子图的高度。默认为"25%"
        subfig_loc                (str) : 子图的位置。默认为'center'
        subfig_bbox              (list) : 子图的位置[横坐标，纵坐标]。默认为[0.5, 0.5]。[0,0]为左下角，[1,1]为右上角
        subfig_bottom_linewidth (float) : 子图底部线条的宽度。默认为1.5
        subfig_left_linewidth   (float) : 子图左侧线条的宽度。默认为1.5
        subfig_xlabel_fontsize    (int) : 子图x轴标签的字体大小。默认为12
        subfig_ylabel_fontsize    (int) : 子图y轴标签的字体大小。默认为12
        subfig_xlabel_fontweight  (str) : 子图x轴标签的字体粗细。默认为normal
        subfig_yticks            (list) : 子图y轴刻度。默认为[0, 5, 10, 15, 20]
        subfig_bar_width        (float) : 子图条形的宽度。默认为0.5
    '''

    primary_cats = kwargs.get('primary_cats', data_in[pry_cat_colname].unique())
    secondary_cats = kwargs.get('secondary_cats', data_in[sec_cat_colname].unique())
    radii = kwargs.get('radii', None)
    radii_value = kwargs.get('radii_value', None)
    ylims = kwargs.get('ylims', None)
    title = kwargs.get('title', None)
    levels_color = kwargs.get('levels_color', None)
    inner_circle_radius = kwargs.get('inner_circle_radius', 0)
    blank_length = kwargs.get('blank_length', 3)
    sort_ascending = kwargs.get('sort_ascending', False)
    sort_by_Total = kwargs.get('sort_by_Total', True)
    bar_linestyle = kwargs.get('bar_linestyle', '--')
    bar_linewidth = kwargs.get('bar_linewidth', 1)
    bar_edgecolor = kwargs.get('bar_edgecolor', 'white')
    bar_alpha = kwargs.get('bar_alpha', 1)
    circle_linestyle = kwargs.get('circle_linestyle', '--')
    circle_linewidth = kwargs.get('circle_linewidth', 1)
    circle_edgecolor = kwargs.get('circle_edgecolor', 'grey')
    circle_alpha = kwargs.get('circle_alpha', 1)
    circle_fill = kwargs.get('circle_fill', False)
    bottom_circle_linestyle = kwargs.get('bottom_circle_linestyle', '-')
    bottom_circle_linewidth = kwargs.get('bottom_circle_linewidth', 2)
    bottom_circle_linecolor = kwargs.get('bottom_circle_linecolor', 'black')
    pry_fontsize = kwargs.get('pry_fontsize', 13)
    sec_fontsize = kwargs.get('sec_fontsize', 13)
    title_fontsize = kwargs.get('title_fontsize', 15)
    circle_label_fontsize = kwargs.get('circle_label_fontsize', 13)
    pry_fontcolor = kwargs.get('pry_fontcolor', 'black')
    sec_fontcolor = kwargs.get('sec_fontcolor', 'black')
    title_fontcolor = kwargs.get('title_fontcolor', 'black')
    bar_label_fontcolor = kwargs.get('bar_label_fontcolor', 'black')
    circle_label_fontcolor = kwargs.get('circle_label_fontcolor', 'black')
    pry_fontweight = kwargs.get('pry_fontweight', 'bold')
    sec_fontweight = kwargs.get('sec_fontweight', 'normal')
    title_fontweight = kwargs.get('title_fontweight', 'normal')
    circle_label_fontweight = kwargs.get('circle_label_fontweight', 'normal')
    legend_on = kwargs.get('legend_on', True)
    legend_label_fontsize = kwargs.get('legend_label_fontsize', 10)
    legend_bbox = kwargs.get('legend_bbox', [0.5, 0.5])
    offset_pry_text = kwargs.get('offset_pry_text', -1.2)
    offset_inner = kwargs.get('offset_inner', -0.2)
    figshow_on = kwargs.get('figshow_on', True)
    
    plot_subfig = kwargs.get('plot_subfig', True)
    subdata_in = kwargs.get('subdata_in', None)
    subfig_width = kwargs.get('subfig_width', "20%")
    subfig_height = kwargs.get('subfig_height', "16%")
    subfig_loc = kwargs.get('subfig_loc', 'center')
    subfig_bottom_linewidth = kwargs.get('subfig_bottom_linewidth', 1.5)
    subfig_left_linewidth = kwargs.get('subfig_left_linewidth', 1.5)
    subfig_xlabel_fontsize = kwargs.get('subfig_xlabel_fontsize', 12)
    subfig_ylabel_fontsize = kwargs.get('subfig_ylabel_fontsize', 12)
    subfig_xlabel_fontweight = kwargs.get('subfig_xlabel_fontweight', 'normal')
    subfig_yticks = kwargs.get('subfig_yticks', None)
    subfig_bar_width = kwargs.get('subfig_bar_width', 0.5)
    subfig_ymin = kwargs.get('subfig_ymin', 0)
    subfig_ymax = kwargs.get('subfig_ymax', 14)
    subfig_title_fontsize = kwargs.get('subfig_title_fontsize', 12)
    subfig_title_fontweight = kwargs.get('subfig_title_fontweight', 'bold')
    
    # add
    panel = kwargs.get('panel', None)
    panellabel_fontsize = kwargs.get('panellabel_fontsize', 15)

    # 计算主要和次要分类的数量
    n_pry = len(primary_cats)
    n_sec = len(secondary_cats)

    # 计算数据级别的最大值和总和的最大值
    max_level = data_in[data_levels].max().max()
    data_in['total'] = data_in[data_levels].sum(axis=1)
    max_sum = data_in['total'].max()
    min_sum = data_in['total'].min()
    
    # 向上取整到最接近的 10 / 2 的倍数
    if ylims is None:
        ymax = math.ceil(max_sum / 2) * 2
        ymin = math.floor(min_sum / 2) * 2
    else:
        ymin, ymax = ylims
    
    print(max_sum, min_sum)
    print(ymax, ymin)
        
    if radii is None:
        radii = np.linspace(ymin, ymax, 5).tolist()
    
    # 计算每个条形的宽度
    width_per_bar = (2 * np.pi) / (n_sec + ((n_pry+1) * blank_length))
    
    # 创建极坐标图
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) 
    # 修改坐标位置，使其充满整个图表
    ax.set_position([0.075, 0.075, 0.85, 0.85])
    # ax.set_position([0.0, 0.035, 0.93, 0.93])
    # 在图片的左上角添加字符a
    plt.text(0.05, 0.95, panel, fontsize=panellabel_fontsize, fontweight='bold', transform=ax.transAxes)

    # 创建颜色级别
    if levels_color is None:
        levels_color = sns.color_palette("Blues_r", len(data_levels))
    else:
        if len(levels_color)<len(data_levels):
            raise ValueError('levels_color must have at least as many colors as data_levels')
            levels_color = sns.color_palette("Blues_r", len(data_levels))

    # 绘制每个数据级别的圆圈,最后一个圆圈不要
    for radius in radii[:-1]:
        circle = plt.Circle((0, 0), radius + inner_circle_radius, transform=ax.transData._b, color=circle_edgecolor, 
                            fill=circle_fill, linestyle=circle_linestyle, linewidth=circle_linewidth, alpha=circle_alpha)
        ax.add_artist(circle)
    
    # 在每个圆圈旁添加文本标签
    # for radius in radii: # '$10^{' +str(radius)+'}$'
    for i, radius in enumerate(radii[:-1]):
        if var == 'loss_usd':
            ax.text(0, radius + inner_circle_radius, '$10^{' +str(int(radii_value[i]))+'}$', ha='center', va='center', color=circle_label_fontcolor,
                fontsize=circle_label_fontsize, rotation_mode='anchor', fontweight=circle_label_fontweight)
        else:
            ax.text(0, radius + inner_circle_radius, str(radii_value[i]), ha='center', va='center', color=circle_label_fontcolor,
                fontsize=circle_label_fontsize, rotation_mode='anchor', fontweight=circle_label_fontweight)
            
    # 初始化起始角度
    angle = width_per_bar * (blank_length+1)

    # 绘制每个主要和次要分类的条形图
    for primary_cat in primary_cats:
        if sort_by_Total:
            primary_cat_data = data_in[data_in[pry_cat_colname] == primary_cat].sort_values(by=['total'], ascending=sort_ascending)
        else:
            primary_cat_data = data_in[data_in[pry_cat_colname] == primary_cat]
        sec_agl = []
        for secondary_cat in primary_cat_data[sec_cat_colname].unique():
            secondary_cat_data = primary_cat_data[primary_cat_data[sec_cat_colname] == secondary_cat]
            bottom = inner_circle_radius
            for j, data_level in enumerate(data_levels):
                value = secondary_cat_data[data_level].sum()
                ax.bar(angle, value, width=width_per_bar, color=levels_color[j], bottom=bottom, 
                       edgecolor=bar_edgecolor, linewidth=bar_linewidth, alpha=bar_alpha, linestyle=bar_linestyle, label=data_level)
                bottom += value
            # 添加与bar平行的text
            text_angle_deg = -np.degrees(angle)+90
            alignment = {'va': 'center', 'ha': 'left'}
            # 检查文本是否位于圆的下半部分
            if text_angle_deg < -90 and text_angle_deg >= -270:
                text_angle_deg += 180
                alignment['ha'] = 'right'
            # 添加次要分类标签
            ax.text(angle, inner_circle_radius, secondary_cat, rotation=text_angle_deg, rotation_mode='anchor', **alignment, 
                    fontsize=sec_fontsize, fontweight=sec_fontweight, color=sec_fontcolor)            
            
            sec_agl.append(angle)
            angle += width_per_bar
        angle += width_per_bar * blank_length
        # 在每个主要分类旁添加文本标签
        if len(sec_agl):
            angles = np.linspace(sec_agl[0]-width_per_bar/2, sec_agl[-1]+width_per_bar/2, 100)
            ax.plot(angles, [inner_circle_radius+offset_inner] * len(angles), color=bottom_circle_linecolor, linewidth=bottom_circle_linewidth)
            center_angle = np.mean(sec_agl)
            text_angle_deg = -np.degrees(center_angle)
            alignment = {'va': 'center', 'ha': 'center'}
            # 检查文本是否位于圆的下半部分
            if text_angle_deg < -90 and text_angle_deg >= -270:
                text_angle_deg += 180
            # 添加主要分类标签
            ax.text(center_angle, inner_circle_radius+offset_pry_text, primary_cat, 
                        rotation=text_angle_deg, **alignment,
                        rotation_mode='anchor', fontsize=pry_fontsize,fontweight=pry_fontweight, color=pry_fontcolor)

    ax.set_ylim(ymin, ymax + inner_circle_radius)
    if title is not None:
        plt.title(title, fontsize=title_fontsize, color=title_fontcolor, fontweight=title_fontweight)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in levels_color[0]]
    labels = data_levels
    if legend_on:
        plt.legend(handles, labels, loc='center', bbox_to_anchor=legend_bbox, fontsize=legend_label_fontsize)
    fig.tight_layout(pad=3.0)
    plt.grid(False)
    plt.axis('off')
    if plot_subfig:
        if subdata_in is None:
            raise ValueError('subdata_in must be provided if plot_subfig is True')
        else:
            ax_inset = inset_axes(ax, width=subfig_width, height=subfig_height, loc='center',
                        bbox_to_anchor=(0, -0.04, 1, 1), bbox_transform=ax.transAxes)
            print(ax_inset,ax)         
            ax_inset.bar(subdata_in['Category'], subdata_in['Value'], yerr=subdata_in['StdDev'], color=levels_color[1], width=subfig_bar_width, capsize=7 )
            ax_inset.spines['top'].set_visible(False)
            ax_inset.spines['right'].set_visible(False)
            ax_inset.spines['bottom'].set_linewidth(subfig_bottom_linewidth) 
            ax_inset.spines['left'].set_linewidth(subfig_left_linewidth) 
            ax_inset.tick_params(axis='x', which='major', labelsize=subfig_xlabel_fontsize)
            ax_inset.tick_params(axis='y', which='major', labelsize=subfig_ylabel_fontsize)
            if subfig_yticks is not None:
                ax_inset.set_yticks(subfig_yticks)
            for label in ax_inset.get_xticklabels():
                label.set_weight(subfig_xlabel_fontweight)
            ax_inset.set_xlim(left=-0.5, right=1.5)
            ax_inset.set_ylim(bottom=subfig_ymin, top=subfig_ymax)
            ax_inset.set_title(subtitle, fontsize=subfig_title_fontsize, fontweight=subfig_title_fontweight)
            
    plt.savefig(f'./fig_pdf/{var}_{v_crop}_{ssp_f}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'./fig_jpg/{var}_{v_crop}_{ssp_f}.jpg', dpi=300, bbox_inches='tight')
    # if figshow_on:
    #     plt.show()
    
    plt.close()


def main():
    color_crops = ['#8397c5','#a9d5d9','#e5f3d6']
    name_crop_mod = ['maize','soy','wheat']
    if ssp_f == 'ssp585':
        crop_dict={'maize':[['#8397c5'],['#c0cae2','#8397c5'],], #'#afbcda'
                'soy':[['#7ab3b8'],['#a9d5d9','#7ab3b8'],], #'#7bb7bc'
                'wheat':[['#c0dba4'],['#e5f3d6','#c0dba4'],]} # '#c5ddac'
    else:
        crop_dict={'maize':[['#c0cae2'],['#c0cae2','#8397c5'],], #'#afbcda'
                'soy':[['#a9d5d9'],['#a9d5d9','#7ab3b8'],], #'#7bb7bc'
                'wheat':[['#e5f3d6'],['#e5f3d6','#c0dba4'],]} # '#c5ddac'
    for i, v_crop in enumerate(name_crop_mod):
        levels_colors = crop_dict[v_crop]
        
        crop_data, global_loss_data = preprocess(v_crop)
        Continents = crop_data['Continent'].unique()
        Continents = ['Asia & Oceania', 'Europe', 'N. America', 'S. America','Africa']
        

        # # 对所有数据级别取对数
        if var == 'loss_usd':
            crop_data[ssp_f] = crop_data[ssp_f]
        else:            
            crop_data[ssp_f] = crop_data[ssp_f] * adj_value  
        global_loss_data['Value']  = global_loss_data['Value'] * adj_value  
        global_loss_data['StdDev'] = global_loss_data['StdDev']/2 * adj_value 
        print(global_loss_data)


        kargs = {'primary_cats'        : Continents,
                 'levels_color'        : levels_colors,
                 'blank_length'        : 3,
                 'pry_fontsize'        : 15,
                 'sec_fontsize'        : 15,
                 'legend_on'           : False,
                 'figshow_on'          : True,
                 'plot_subfig'         : True,
                 'subdata_in'          : global_loss_data,
                 'subfig_xlabel_fontweight' : 'bold',
                 'subfig_ymin'         : 0,#ylims_dict[v_crop][0],
                 'subfig_ymax'         : ymax_value,#ylims_dict[v_crop][1],
                 'subfig_title_fontsize' : 19,
                 'circle_label_fontsize' : 14,
                 'subfig_xlabel_fontsize' : 15,
                 'panellabel_fontsize'    : 24,
                 'ylims'                : [0, 8],
                 'inner_circle_radius'  : 8,
                 'radii_value'          : radii_value,
                 'panel'                : subpanel_value[i],
                #  'title'                : v_crop,
                }
        
        # --------  crop_data 必须经过预处理，添加了Continent列
        # Radial_histogram(v_crop, crop_data, 'Continent', 'Area', ssp_names, ylims=[0, 30], radii=[0, 5, 10, 15, 20], inner_circle_radius=30, 
        #                 blank_length=3, primary_cats=Continents, pry_fontsize=15, sec_fontsize=12)
        # Radial_histogram(v_crop, crop_data, 'Continent', 'Area', ssp_name,  radii=[0, 2, 4, 6, 8, 10], inner_circle_radius=14, levels_color=levels_colors,
        #         blank_length=3, primary_cats=Continents, pry_fontsize=15, sec_fontsize=12, legend_on=False, figshow_on=True, subdata_in=global_loss_data, plot_subfig=True)
        Radial_histogram(v_crop, crop_data, 'Continent', 'Area', [ssp_f], **kargs)

if __name__ == "__main__":
    
    os.makedirs('./fig_pdf', exist_ok=True)
    os.makedirs('./fig_jpg', exist_ok=True)

    for ssp_f in ['ssp585']:
        f_data = f'./data/'

        name_var = ['loss_usd','pergdp_loss_usd', 'perpop_loss_production']
        for var in name_var[2:3]:
            max_dict   = {'pergdp_loss_usd':4,'perpop_loss_production':88,'loss_usd':12}      # 对每个国家数值进行标准化，/max_value*8
            max_value  = max_dict[var]
            adj_dict   = {'pergdp_loss_usd':1000,'perpop_loss_production':1000,'loss_usd':1/10**9}   # 中间图的调整值, *adj_value
            adj_value  = adj_dict[var]
            ymax_dict  = {'pergdp_loss_usd':0.12,'perpop_loss_production':10,'loss_usd':8}    # 中间图y轴的最大值
            ymax_value = ymax_dict[var]
            subtitle_dict = {'pergdp_loss_usd':'Loss as a share of GDP\n(\u2030)','perpop_loss_production':'Per capita loss\n(ton)','loss_usd':'Total loss\n(USD)'}
            subtitle      = subtitle_dict[var]
            radii_dict    = {'pergdp_loss_usd':[0, 1, 2, 3, 4],'perpop_loss_production':[0, 22, 44, 66, 88],'loss_usd':[0, 3, 6, 9, 12]}
            radii_value   = radii_dict[var]
            subpanel_dict = {'pergdp_loss_usd':['d     Maize','e      Soybean','f      Wheat'],
                            'perpop_loss_production':['g     Maize','h      Soybean','i      Wheat'],
                            'loss_usd':['a     Maize','b      Soybean','c      Wheat']}
            subpanel_value= subpanel_dict[var]

            main()
    
    