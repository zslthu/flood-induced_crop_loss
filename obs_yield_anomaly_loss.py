import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# revise based on : Li, Y., Guan, K., Schnitkey, G. D., DeLucia, E., & Peng, B. (2019). 
# Excessive rainfall leads to maize yield loss of a comparable magnitude to extreme drought in the United States.
# Global Change Biology, 25(7), 2325–2337. https://doi.org/10.1111/gcb.14628
def cal_yield_anomaly(data_combined, crop, fitting_type='linear',rerun=False):
    
    if rerun:
        combined_sample = data_combined[['FIPS','State','Year','Yield','Area']].dropna().set_index(['FIPS','Year'])
        # print(combined_sample)
        s = combined_sample.unstack('FIPS')['Yield'].shape  # size, year by FIPS
        # print(s)

        if fitting_type=='linear':
            formula_txt = "Yield ~ Year"
            B = np.zeros([s[1],6]) # a,b,n, three parameters (intercept, slope, and sample number) for linear trend of yield and area, 
            trend_para = pd.DataFrame(B, index=combined_sample.unstack('FIPS')['Yield'].columns, \
                                    columns=['Yield_intercept','Yield_slope','Yield_N',
                                            'Area_intercept','Area_slope','Area_N'])
            
        if fitting_type=='quadratic':
            formula_txt = "Yield ~ Year + np.power(Year, 2)"
            B = np.zeros([s[1],7]) # a,b,n, four parameters (intercept, slope_1,slope_2, and sample number) for linear trend of yield and area, 
            trend_para = pd.DataFrame(B, index=combined_sample.unstack('FIPS')['Yield'].columns, \
                                    columns=['Yield_intercept','Yield_slope1','Yield_slope2','Yield_N',
                                            'Area_intercept','Area_slope','Area_N'])

        # estimate linear trend for each column (FIPS)
        for i in range(s[1]): #s[1]
            # First make yield anomaly
            # print(i)
            P_bin = combined_sample.unstack('FIPS')['Yield'].iloc[:,i].to_frame('Yield').reset_index()
            # print(P_bin)
            mod_fit = smf.ols(formula=formula_txt, data=P_bin).fit()
            # print(mod_fit)
            if fitting_type == 'linear':
                # B[i,0],B[i,1],B[i,2]= mod_fit.params[0], mod_fit.params[1], P_bin['Yield'].dropna().shape[0]
                B[i,0],B[i,1],B[i,2]= mod_fit.params.iloc[0], mod_fit.params.iloc[1], P_bin['Yield'].dropna().shape[0]
            if fitting_type=='quadratic':
                # B[i,0],B[i,1],B[i,2], B[i,3]=mod_fit.params[0],mod_fit.params[1],mod_fit.params[2],P_bin['Yield'].dropna().shape[0]
                B[i,0],B[i,1],B[i,2], B[i,3]=mod_fit.params.iloc[0],mod_fit.params.iloc[1],mod_fit.params.iloc[2],P_bin['Yield'].dropna().shape[0]

            # Second make area anomaly
            P_bin2 = combined_sample.unstack('FIPS')['Area'].iloc[:,i].to_frame('Area').reset_index()
            mod_fit2 = smf.ols(formula="Area ~ Year", data=P_bin2).fit()
            if fitting_type == 'linear':
                # B[i,3],B[i,4],B[i,5]= mod_fit2.params[0], mod_fit2.params[1], P_bin2['Area'].dropna().shape[0]
                B[i,3],B[i,4],B[i,5]= mod_fit2.params.iloc[0], mod_fit2.params.iloc[1], P_bin2['Area'].dropna().shape[0]
            if fitting_type=='quadratic':
                # B[i,4],B[i,5],B[i,6]= mod_fit2.params[0], mod_fit2.params[1], P_bin2['Area'].dropna().shape[0]
                B[i,4],B[i,5],B[i,6]= mod_fit2.params.iloc[0], mod_fit2.params.iloc[1], P_bin2['Area'].dropna().shape[0]    


        yield_ana_sample = combined_sample.unstack('FIPS')['Yield'].copy()
        area_ana_sample = combined_sample.unstack('FIPS')['Area'].copy()

        # get anomaly by array multiplication through broadcasting
        year_start = combined_sample.index.get_level_values(1).min()
        year_end = combined_sample.index.get_level_values(1).max()
        num_year = year_end - year_start + 1
        # print(year_start,year_end,num_year)

        if fitting_type == 'linear':
            array_yield_ana = yield_ana_sample.values - \
                np.array([np.arange(year_start, year_end + 1),] * s[1]).T \
                * np.array([trend_para.T.loc['Yield_slope'].values,] * num_year) \
                - np.array([trend_para.T.loc['Yield_intercept'].values,] * num_year)
        if fitting_type=='quadratic':
            array_yield_ana = yield_ana_sample.values - \
                np.array([np.arange(year_start, year_end + 1),] * s[1]).T \
                * np.array([trend_para.T.loc['Yield_slope1'].values,] * num_year) \
                - np.power(np.array([np.arange(year_start, year_end + 1),] * s[1]).T, 2) \
                * np.array([trend_para.T.loc['Yield_slope2'].values,] * num_year) \
                - np.array([trend_para.T.loc['Yield_intercept'].values,] * num_year)

        yield_ana_sample.iloc[:,:] = array_yield_ana

            
        array_area_ana = area_ana_sample.values - \
            np.array([np.arange(year_start, year_end + 1),] * s[1]).T \
            * np.array([trend_para.T.loc['Area_slope'].values,] * num_year) \
            - np.array([trend_para.T.loc['Area_intercept'].values,] * num_year)    
            
        area_ana_sample.iloc[:,:] = array_area_ana

        # append anomaly to yield data
        combined_sample = combined_sample.reset_index(). \
        merge(yield_ana_sample.stack().reset_index().rename(columns={0:'Yield_ana'})).\
        merge(area_ana_sample.stack().reset_index().rename(columns={0:'Area_ana'}))

        # save for reuse
        combined_sample.to_csv(f'{output_dir}/{crop}_yield_area_anomaly_{fitting_type}.csv', index=False)
        trend_para.to_csv(f'{output_dir}/{crop}_yield_area_trend_para_{fitting_type}.csv', index=False)
        print(f'{crop}_yield_area_anomaly_{fitting_type}.csv save!')

    else:
        combined_sample = pd.read_csv(f'{output_dir}/{crop}_yield_area_anomaly_{fitting_type}.csv', dtype={'FIPS':str,'State':str})
        trend_para = pd.read_csv(f'{output_dir}/{crop}_yield_area_trend_para_{fitting_type}.csv', dtype={'FIPS':str,'State':str})

    return combined_sample, trend_para

def cal_clim_bin(yield_sample,prec_gs,fips):
    P = prec_gs[['Year',fips]].copy()    
    P.columns = ['Year','Prec'] 

    P['Prec_percentile']=(stats.rankdata(P['Prec'], method='average')*2-1)/(P['Prec'].shape[0]*2)

    P_bin = P.copy()
    c = P_bin.index <= 2022
    v_mean = P_bin[c]['Prec'].mean()
    v_std = P_bin[c]['Prec'].std()

    prec_bin_sigma = [v_mean + i * v_std for i in np.arange(-3.5,3.6,0.5)]

    prec_bin_rank = np.arange(0,1.0001,0.05)

    bin_means1, bin_edges1, binnumber1 = stats.binned_statistic(P_bin['Prec_percentile'], P_bin['Prec_percentile'], 'mean', bins=prec_bin_rank)
    bin_means2, bin_edges2, binnumber2 = stats.binned_statistic(P_bin['Prec'], P_bin['Prec'], 'mean', bins=prec_bin_sigma)

    P_bin['Prec_rank_bin'] =  binnumber1
    P_bin['Prec_sigma_bin'] = binnumber2
    P_bin['Prec_to_sd'] = (P_bin['Prec'] - v_mean)/v_std

    data_bin = P_bin.merge(yield_sample[yield_sample['FIPS']==fips], on='Year', how='left')
    # print(data_bin)

    return data_bin


def main_process(crop='corn'):
    # # ==== load yield data
    files = os.listdir(usda_dir)
    csv_filename = [f for f in files if f.startswith(f'{crop}_both')] 
    data_combined = pd.read_csv(f'{usda_dir}/{csv_filename[0]}',dtype={'FIPS':str,'State':str})
    data_combined.rename(columns={'Areaharvested':'Area'},inplace=True)
    data_combined = data_combined[(data_combined['Year']>=start_year) & (data_combined['Year']<=end_year)]

    # # ==== Add yield and area anomaly
    fitting_type = 'linear'
    yield_sample, trend_para = cal_yield_anomaly(data_combined, crop, fitting_type=fitting_type, rerun=True)

    # # ==== load climate data
    prec_gs = pd.read_csv(clim_dir+'/ppt_growing_season.csv')
    prec_gs.columns = ['Year'] + [str(i).zfill(5) for i in prec_gs.columns[1:]]

    # # ==== merge yield and climate data
    F1 = yield_sample['FIPS'].unique()
    F2 = prec_gs.columns.values    
    fips_all = np.intersect1d(F1, F2)
    # print(fips_all.shape)

    frame = [cal_clim_bin(yield_sample, prec_gs, fips) for fips in fips_all]
    bin_yield = pd.concat(frame)
    bin_yield = bin_yield.dropna()  
    # print(bin_yield)

    # Calculate reduction percent relative to the trend term
    bin_yield['Yield_ana_to_yield'] = bin_yield['Yield_ana']/(bin_yield['Yield'] - bin_yield['Yield_ana'])
    bin_yield['Production_ana'] = bin_yield['Area'] * bin_yield['Yield_ana']
    bin_yield['Yield_ana_to_yield_area'] = bin_yield['Area'] * bin_yield['Yield_ana_to_yield']

    # Add normal condition anomaly
    con_normal = (bin_yield['Prec_sigma_bin']>=6)&(bin_yield['Prec_sigma_bin']<=8)
    bin_yield = bin_yield.join(bin_yield[con_normal].groupby(['FIPS','State']).mean()['Yield_ana_to_yield'].to_frame('Yield_ana_to_yield_normal'), how='left',on=['FIPS','State'])
    # bin_yield = bin_yield.join(bin_yield[con_normal].groupby(['FIPS','State']).mean()['Yield_ana_to_yield'].to_frame('Yield_ana_to_yield_normal'), how='left',on='FIPS')
    bin_yield['Yield_ana_to_yield_normal_diff'] = bin_yield['Yield_ana_to_yield'] - bin_yield['Yield_ana_to_yield_normal']
    # print(bin_yield)

    # # save result
    fn = f'{output_dir}/{crop}_climbin_yield_anomaly_{fitting_type}.csv'
    bin_yield.to_csv(fn,index=False)    
    print(f'{fn} save!')

#=========== add loss ratio =========================================
def main_add_loss(crop):
    # ==== Load loss ratio data
    loss_ratio_cause = pd.read_csv(f'{rma_dir}/{crop}_loss_ratio_cause.csv', dtype={'FIPS': str})
    loss_ratio_cause.rename(columns={'Commodity Year': 'Year'}, inplace=True)
    # print(loss_ratio_cause)

    # ==== change cause names
    cause_names = loss_ratio_cause['Damage Cause Description'].value_counts()
    # print(cause_names,'\n\n')

    # Index loss ratios for quick access
    indexed_loss_ratios = loss_ratio_cause.set_index(['FIPS', 'Year', 'Damage Cause Description'])

    # ==== Load yield data
    bin_yield = pd.read_csv(f'{output_dir}/{crop}_climbin_yield_anomaly_{fitting_type}.csv', dtype={'FIPS': str})   
        
    def get_loss_ratio(row, indexed_loss_ratios, cause_txt,col_txt):
        try:
            return indexed_loss_ratios.loc[(row['FIPS'], row['Year'], cause_txt), col_txt]
        except KeyError:
            return np.nan
    
    # Function to add loss ratios
    def add_loss_ratios(group):
        fips = group['FIPS'].iloc[0]
        # print(indexed_loss_ratios)
        if fips in indexed_loss_ratios.index:
            for cause_txt in cause_names_sele:
                group[f'loss_ratio_{cause_txt}'] = group.apply(get_loss_ratio, args=(indexed_loss_ratios, cause_txt,'Loss ratio by cause'), axis=1)
                group[f'indemnity_{cause_txt}'] = group.apply(get_loss_ratio, args=(indexed_loss_ratios, cause_txt,'Indemnity Amount by cause'), axis=1)
        else:
            for cause_txt in cause_names_sele:
                group[f'loss_ratio_{cause_txt}'] = np.nan
                group[f'indemnity_{cause_txt}'] = np.nan
        return group

    bin_yield = bin_yield.groupby('FIPS').apply(add_loss_ratios).reset_index(drop=True)

    # save result
    fn = f'{output_dir}/{crop}_climbin_yield_anomaly_loss_{fitting_type}_indemnity.csv'
    bin_yield.to_csv(fn, index=False)
    print(f'{fn} saved!')
   

#================================================================== main process
usda_dir   = '../output/usda_yield_data'
rma_dir    = '../output/usda_rma_data'
clim_dir   = '../data'
output_dir = '../output/obs_data'
os.makedirs(output_dir, exist_ok=True)

start_year = 1950
end_year   = 2016

fitting_type = 'linear'

with_loss_ratio = True
cause_names_sele = ['Drought','Excess Moisture/Precip/Rain','Flood']

name_crop = ['corn','soybeans','wheat']
for ic in range(3): 
    crop = name_crop[ic]
    # main_process(crop)

    if with_loss_ratio:
        main_add_loss(crop) 

