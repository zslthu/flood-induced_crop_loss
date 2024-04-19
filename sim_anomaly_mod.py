import glob
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import numpy as np
import os
import sys



def load_mod_data(model_name, climate_name, anomaly=False):

    if anomaly:  
        if os.path.exists(f_dir_output) == False:
            os.makedirs(f_dir_output)
        end_string = f'{model_name}_{climate_name}_hist_default_noirr_yield_{crop_mod[0:3]}_annual_anomaly_{yr_start}_{yr_end_model[climate_name]}.nc'
        ##=== for isimip
        # end_string = f'{model_name}_{climate_name}_{ssp}_default_noirr_yield_{crop_mod[0:3]}_annual_anomaly_{yr_start}_{yr_end}.nc'
        fn = f'{f_dir_output}/{end_string}'
        ds = xr.open_dataset(fn, decode_times=False)

    else:
        if not flag_adj:
            fn_tmp = f'{f_dir_mod}/{model_name}/{model_name}.{crop_mod}/{climate_name}'
            ##=== for isimip
            # fn_tmp = f'{f_dir_mod}/{model_name}/{climate_name}'
        else:
            fn_tmp = f'{f_dir_mod}/{model_name}'
        
        end_string = f'{model_name}*{climate_name}*_hist_default_noirr_yield_*{crop_mod[0:3]}*'
        ##=== for isimip 
        # end_string = f'{model_name}_{climate_name}_w5e5_{ssp}_2015soc_default_yield-{crop_mod[0:3]}*'
        fn = glob.glob(f'{fn_tmp}/{end_string}')[0]
        yr_s = int(fn.split('_')[-2])
        yr_e = int(fn.split('_')[-1].split('.')[0])
        ds = xr.open_dataset(fn, decode_times=False)
        ds['time'] = np.arange(yr_s,yr_e+1,1)    
          
    return ds


def get_yield_ana(ds_mod, var_name, yr_start, yr_end):

    mask_mod = ds_mod[var_name].sum(axis=0).values
    mask = mask_mod!=0
    idx = np.argwhere(mask)
        
    # Get slope and intercept for trend
    X = np.arange(yr_start,yr_end+1)
    X = sm.add_constant(X)

    array_slope1 = np.zeros([360,720])
    array_intercept1 = np.zeros([360,720])

    for n in range(idx.shape[0]):
        lat_n = idx[n][0]
        lon_n = idx[n][1]

        y1 = ds_mod[var_name][:,lat_n,lon_n].values 

        mod_fit1 = sm.OLS(y1, X, missing='drop').fit()

        array_intercept1[lat_n,lon_n], array_slope1[lat_n,lon_n] = \
            mod_fit1.params[0], mod_fit1.params[1]
                    
    # Get anomaly 
    n_yr = yr_end - yr_start + 1
    array_year = np.zeros([n_yr,360,720])
    for y in range(yr_start,yr_end+1):
        array_year[y-yr_start:,:] = y

    array_trend = np.zeros([n_yr,360,720])

    for y in range(yr_start,yr_end+1):
        array_trend[y-yr_start:,:] = array_year[y-yr_start:,:] * array_slope1 + array_intercept1

    array_ana = ds_mod[var_name].values - array_trend
    
    mask_3d = np.broadcast_to(mask, array_ana.shape)
    array_ana[~mask_3d] = np.nan
    array_trend[~mask_3d] = np.nan
    
    return array_ana, array_trend


def save_anomaly_mod(name_model,climate_name):
    yr_end = yr_end_model[climate_name]
    
    for m in name_model:

        var_name = f'yield_{crop_mod[0:3]}'
        ##=== for isimip
        # var_name = f'yield-{crop_mod[0:3]}-noirr'

        ds_mod = load_mod_data(m, climate_name)
        data_mod = ds_mod.sel(time=range(yr_start,yr_end+1))
        array_ana, array_trend = get_yield_ana(data_mod, var_name, yr_start, yr_end)

        yield_mod = ds_mod.sel(time=range(yr_start,yr_end+1))[var_name].values

        ds = xr.Dataset({'yield_ana': (['time', 'lat', 'lon'], array_ana),
                         'yield_trend': (['time', 'lat', 'lon'], array_trend),
                         'yield': (['time', 'lat', 'lon'], yield_mod),
                               },
                        coords={'lon': ds_mod.lon,
                                'lat': ds_mod.lat,
                                'time': np.arange(yr_start,yr_end+1,1)
                                 })

        end_string = f'{m}_{climate_name}_hist_default_noirr_yield_{crop_mod[0:3]}_annual_anomaly_{yr_start}_{yr_end}.nc'
        ##=== for isimip
        # end_string = f'{m}_{climate_name}_{ssp}_default_noirr_yield_{crop_mod[0:3]}_annual_anomaly_2015_2100.nc'

        ds.to_netcdf(f'{f_dir_output}/{end_string}')
        print(f'{end_string} saved!')


#=========================================================== 
if __name__ == '__main__':


    # # simulation parameters
    name_crop_obs  = ['corn','soybeans','wheat']
    name_crop_mod  = ['maize','soy','wheat']
    name_model     = ['cgms-wofost','lpj-guess','clm-crop','lpjml','epic-iiasa','gepic', 'orchidee-crop','pdssat','papsim','pegasus']
    name_climate   = ['wfdei.gpcc','agmerra']

    yr_start       = 1981
    yr_end_model   = {'agmerra': 2010, 'wfdei.gpcc': 2009}
   
    flag_adj  = 0
    # # model data
    if not flag_adj:
        f_dir_mod     = '/tera05/zhangsl/GGCMI_phase1_unzipped'
        f_dir_output  = f'../output/anomaly_mod/agmip_org/'
    else:
        f_dir_mod = '../output/AgMIP_adj'
        f_dir_output = f'../output/anomaly_mod/agmip_adj/'    
    os.makedirs(f_dir_output, exist_ok=True)

    # ##=== for isimip
    # # # simulation parameters
    # name_crop_mod  = ['maize','soy','wheat']
    # name_model     = ['crover','epic-iiasa', 'ldndc', 'lpj-guess', 'lpjml', 'pdssat', 'pepic', 'promet', 'simplace-lintul5'] 
    # name_climate   = ['gfdl-esm4','ukesm1-0-ll','mri-esm2-0','mpi-esm1-2-hr','ipsl-cm6a-lr']

    # yr_start       = 2015
    # yr_end         = 2100
   
    # ssp = 'ssp585'
    # flag_adj  = 1
    # # # model data
    # if not flag_adj:
    #     f_dir_mod     = f'/tera07/zhangsl/lianghb21/ISMIP/ISMIP_3b/crop_{ssp}/'
    #     f_dir_output  = f'../output/anomaly_mod/isimip_org/{ssp}'
    # else:
    #     f_dir_mod    = f'../output/ISIMIP_adj/{ssp}'
    #     f_dir_output = f'../output/anomaly_mod/isimip_adj/{ssp}'    
    # os.makedirs(f_dir_output, exist_ok=True)

    #====================================================================
    for ic in range(1):
        crop_mod = name_crop_mod[ic]

        for climate_name in name_climate[0:1]:
            save_anomaly_mod(name_model,climate_name)

