import glob
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import numpy as np
import os
import sys
from sim_anomaly_mod import get_yield_ana


def save_yield_obs_anamaly(f_dir_obs):
    
    ds_obs = xr.open_dataset(f_dir_obs)
    array_ana, array_trend = get_yield_ana(ds_obs, 'yield', yr_start=1981, yr_end=2010)

    ds_obs_ana = xr.Dataset({'yield_ana': (['time', 'lat', 'lon'], array_ana),
                             'yield_trend': (['time', 'lat', 'lon'], array_trend),
                             'yield': (['time', 'lat', 'lon'], ds_obs['yield'].values),
                             'area': (['time', 'lat', 'lon'], ds_obs['area'].values)
                           },
                    coords={'lon': ds_obs.lon.values,
                            'lat': ds_obs.lat.values,
                            'time': ds_obs.time.values
                             })

    ds_obs_ana.to_netcdf(f_dir_obs_out)
    print(f'{f_dir_obs_out} saved!')


def xarray2dataframe(da, name):
    return da.to_series().dropna().rename(name).to_frame().reset_index()


def grid_to_csv(climate_name):

    ds_obs = xr.open_dataset(f_dir_obs_out)   
    ds_obs['yield_ana_to_yield'] = ds_obs['yield_ana'] / ds_obs['yield'].mean(dim='time')   

    ds_obs_clim = xr.open_dataset(f_dir_obs)

    t = xarray2dataframe(ds_obs['yield_ana_to_yield'], 'obs')
    t1 = xarray2dataframe(ds_obs['area'], 'Area')
    t2 = xarray2dataframe(ds_obs_clim['Prec_sigma_bin'], 'Prec_sigma_bin')
    t = t.merge(t1, how='left').merge(t2, how='left')

    for m in name_model:
        end_string = f'{m}_{climate_name}_hist_default_noirr_yield_{crop_mod[0:3]}_annual_anomaly_{yr_start}_{yr_end_model[climate_name]}.nc'
        fn = f'{f_dir_output}/{end_string}'
        ds = xr.open_dataset(fn, decode_times=False)
 
        ds['yield_ana_to_yield'] = ds['yield_ana'] / ds['yield'].mean(dim='time')  
        t3 = xarray2dataframe(ds['yield_ana_to_yield'], m)
        t = t.merge(t3, on=['lat','lon','time'], how='left')

    fn = f'{f_dir_csv}/obs_mod_{crop_mod}_noirr_{climate_name}.csv'
    t.to_csv(fn, index=None)
    print(f'{fn} saved!')


#=========================================================== 
if __name__ == '__main__':

    # # simulation parameters
    name_crop_obs  = ['corn','soybeans','wheat']
    name_crop_mod  = ['maize','soy','wheat']
    name_model     = ['cgms-wofost','lpj-guess','clm-crop','lpjml','epic-iiasa','gepic', 'orchidee-crop','pdssat','papsim','pegasus']
    name_climate   = ['wfdei.gpcc','agmerra']

    yr_start       = 1981
    yr_end_model   = {'agmerra': 2010, 'wfdei.gpcc': 2009}   

    flag_adj  = 1
    if not flag_adj:
        f_dir_mod     = '/tera05/zhangsl/GGCMI_phase1_unzipped'        
        f_dir_output  = f'../output/anomaly_mod/agmip_org/'
        f_dir_csv     = '../output/obs_mod_csv/yield_org'
    else:
        f_dir_mod = '../output/AgMIP_adj'
        f_dir_output = f'../output/anomaly_mod/agmip_adj/'
        f_dir_csv     = '../output/obs_mod_csv/yield_adj'
    
    os.makedirs(f_dir_csv, exist_ok=True)
    os.makedirs(f_dir_output, exist_ok=True)

    # # obs dir
    f_dir_o       = '../output/anomaly_obs/'
    #====================================================================
    for ic in range(3):
        #======= obs ========
        crop_obs = name_crop_obs[ic]
        crop_mod = name_crop_mod[ic]
        # # obs data
        f_dir_obs     = f'{f_dir_o}/{crop_obs}_grid_climbin_yield_1981_2010_05deg.nc'
        f_dir_obs_out = f'{f_dir_o}/{crop_obs}_grid_climbin_yield_anomaly_1981_2010_05deg.nc'

        # obs anamaly    
        save_yield_obs_anamaly(f_dir_obs)
        for climate_name in name_climate:
            grid_to_csv(climate_name)


