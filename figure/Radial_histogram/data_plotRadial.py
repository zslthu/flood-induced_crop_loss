
import pandas as pd
import numpy as np
import os

name_crop_mod = ['maize','soy','wheat']
model_names   = ['crover','epic-iiasa', 'ldndc', 'lpj-guess', 'lpjml', 'pdssat', 'pepic', 'promet', 'simplace-lintul5'] 
climate_names = ['gfdl-esm4','ukesm1-0-ll','mri-esm2-0','mpi-esm1-2-hr','ipsl-cm6a-lr']
name_crop     = ['maize','soy','wheat','wheat','wheat']
ssp_names     = ['ssp126','ssp585']

yr_start = 2021
yr_end   = 2100
loss_type  = 'usd'
add        = 'pergdp_'
# loss_type  = 'production'
# add        = 'perpop_'

f_loss_save = '../../../output/loss_glb_ssp/ssp_adj'
f_data_save = f'./data'
os.makedirs(f_data_save, exist_ok=True)

start_string_all = [(model+'_'+climate) for model in model_names for climate in climate_names]

for iv, v_crop in enumerate(name_crop_mod):
    data_loss_global = pd.DataFrame(columns=['M49','Country name','Item','model','climate'])
    data_loss_global['model'] = [start_string.split('_')[0] for start_string in start_string_all]
    data_loss_global['climate'] = [start_string.split('_')[1] for start_string in start_string_all]
    data_loss_global['M49'] = 0
    data_loss_global['Country name'] = 'Global'
    data_loss_global['Item'] = v_crop 
    
    for ssp in ssp_names:
        ii = -1
        for start_string in start_string_all:
            f_loss = f'{f_loss_save}/{ssp}'
            data_loss_usd = pd.read_csv(f'{f_loss}/loss_{loss_type}/{add}loss_{loss_type}_{v_crop}_{start_string}_{ssp}.csv')
            col_start = data_loss_usd.columns.get_loc(f'Y{yr_start}')
            col_end = data_loss_usd.columns.get_loc(f'Y{yr_end}') 

            # print(data_loss_usd)
            ii += 1 
            data_loss_global.loc[ii,ssp] = data_loss_usd.iloc[0, col_start:col_end+1].mean()

    data_loss_global.to_csv(f'{f_data_save}/global_{add}loss_{loss_type}_{v_crop}.csv', index=False)

start_string = 'ensemble'
for iv, v_crop in enumerate(name_crop_mod):
    # pd to save results
    f_loss = f'{f_loss_save}/ssp126'
    data_loss_usd = pd.read_csv(f'{f_loss}/loss_{loss_type}/{add}loss_{loss_type}_{v_crop}_ensemble_ssp126.csv')
    data_loss_ssp = data_loss_usd[['M49','Country name','Item']].copy()
    col_start = data_loss_usd.columns.get_loc(f'Y{yr_start}')
    col_end = data_loss_usd.columns.get_loc(f'Y{yr_end}') 

    for ssp in ssp_names:
        f_loss = f'{f_loss_save}/{ssp}'
        data_loss_usd = pd.read_csv(f'{f_loss}/loss_{loss_type}/{add}loss_{loss_type}_{v_crop}_{start_string}_{ssp}.csv')
        # print(data_loss_usd)

        data_loss_ssp[ssp] = data_loss_usd.iloc[:, col_start:col_end+1].mean(axis=1)

    data_loss_ssp.to_csv(f'{f_data_save}/{add}loss_{loss_type}_{v_crop}.csv', index=False)

