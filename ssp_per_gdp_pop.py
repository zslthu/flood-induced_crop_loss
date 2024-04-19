import pandas as pd
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def process_each_model(start_string): 
    ## per gdp    
    f_loss_usd = f'{f_loss}/loss_usd'
    fname = f'loss_usd_{v_crop}_{start_string}_{ssp}.csv'   
    data_loss_usd = pd.read_csv(f'{f_loss_usd}/{fname}')
    # print(data_loss_usd)
    for i in range(len(data_loss_usd)):
        gdp = data_gdp.loc[data_gdp['M49'] == data_loss_usd.loc[i,'M49']]
        # print(gdp['avg_gdp'].values)
        data_loss_usd.iloc[i,3::] = data_loss_usd.iloc[i,3::].values / gdp['avg_gdp'].values
    # print(data_loss_usd)
    data_loss_usd.to_csv(f'{f_loss_usd}/pergdp_{fname}', index=False)

    ## per pop
    f_loss_production = f'{f_loss}/loss_production'
    fname = f'loss_production_{v_crop}_{start_string}_{ssp}.csv' 
    data_loss_production = pd.read_csv(f'{f_loss_production}/{fname}')
    # print(data_loss_production)
    for i in range(len(data_loss_production)):
        pop  = data_pop_all.loc[data_pop['M49'] == data_loss_production.loc[i,'M49']]
        data_loss_production.iloc[i,3::] = data_loss_production.iloc[i,3::].values / pop.iloc[0,2::].values
    # print(data_loss_production)
    data_loss_production.to_csv(f'{f_loss_production}/perpop_{fname}', index=False)

#====== model information
name_crop_mod = ['maize','soy','wheat']
model_names   = ['crover','epic-iiasa', 'ldndc', 'lpj-guess', 'lpjml', 'pdssat', 'pepic', 'promet', 'simplace-lintul5'] 
climate_names = ['gfdl-esm4','ukesm1-0-ll','mri-esm2-0','mpi-esm1-2-hr','ipsl-cm6a-lr']
name_crop     = ['maize','soy','wheat']
name_ssp      = ['ssp126','ssp585']

f_loss_save = f'../output/loss_glb_ssp/ssp_adj'

#====== load GDP data
dir_gdp = f'../data/GDP_dataset/GDP.csv'
data_gdp = pd.read_csv(dir_gdp)
data_gdp['avg_gdp'] = data_gdp.iloc[:,2:].mean(axis=1)   

for ssp in name_ssp[1:2]:
    f_loss = f'{f_loss_save}/{ssp}'

    #====== load population data
    dir_pop = f'../data/ssp_population/{ssp}.csv'
    data_pop = pd.read_csv(dir_pop)
    data_pop_all = data_pop.iloc[:,0:2].copy()
    data_pop_all['2015'] = data_pop['2020'] 
    for year in range(2016,2101):
        if year%5 == 0:
            year_nearest = year
        else:
            year_nearest = (year//5+1)*5
        data_pop_all[f'{year}'] = data_pop[f'{year_nearest}']

    #====== load loss data
    start_string_all = [(model+'_'+climate) for model in model_names for climate in climate_names]
    for v_crop in name_crop_mod:        
        pool = multiprocessing.Pool(processes=32)
        pool.map(process_each_model, start_string_all)
        pool.close()
        pool.join()
        print(f'{v_crop} done')            
                        




