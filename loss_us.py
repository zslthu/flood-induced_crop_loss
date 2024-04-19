import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from get_county_latlon import get_county_latlon
import os

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def find_fips_vectorized(row_b, data_a):
    distances = haversine_np(data_a['lat'].values, data_a['lon'].values, row_b['lat'], row_b['lon'])
    closest_index = np.argmin(distances)
    return data_a.iloc[closest_index]['FIPS']

def xarray2dataframe(da, name):
    return da.to_series().dropna().rename(name).to_frame().reset_index()


np.seterr(all='ignore')

# # model information
name_crop_obs = ['corn','soybeans','wheat']
name_crop_mod = ['maize','soy','wheat']
model_names   = ['cgms-wofost','lpj-guess','clm-crop','lpjml','epic-iiasa','gepic','orchidee-crop','pdssat','papsim','pegasus']
climate_names = ['agmerra','wfdei.gpcc']

yr_start = 1981
dict_year = {'wfdei.gpcc':2009,'agmerra':2010}

# # dir for crop yield
dir_mod_org  = '../output/anomaly_mod/agmip_org/'
dir_mod_adj  = '../output/anomaly_mod/agmip_adj/'

# # # dir for crop price and country code
dir_country = '../data/crop_price/M49_30mn.nc'
data_country = xr.open_dataset(dir_country)['Band1'].values

county_info = get_county_latlon(rerun=False)

# # dir for crop area
dir_crop  = '../data/crop_area/crop_area_30mn.nc'
data_area = xr.open_dataset(dir_crop)

f_indemnity = '../output/obs_mod_csv/indemnity/'
os.makedirs(f_indemnity, exist_ok=True)

# # main
for ic in range(3):
    crop_mod   = name_crop_mod[ic]
    crop_obs   = name_crop_obs[ic]

    # # obs data
    f_dir_obs  = f'../output/anomaly_obs/{crop_obs}_grid_climbin_yield_1981_2010_05deg.nc'
    ds_obs     = xr.open_dataset(f_dir_obs)
    var_names  = list(ds_obs.data_vars.keys())
    data_obs   = ds_obs[var_names[0]].to_series().dropna().to_frame().reset_index()
    for var in var_names[1::]:
        data_obs_temp = ds_obs[var].to_series().dropna().to_frame().reset_index()
        data_obs      = data_obs.merge(data_obs_temp, on=['time', 'lat', 'lon'], how='outer')

    # # [var A] read crop area for each crop
    crop_area = data_area[crop_mod].values

    # # [var P] read crop price for each crop
    dir_price = f'../data/crop_price/crop_price_fillna_{crop_mod}.csv'
    data_price_crop = pd.read_csv(dir_price)
    idx_price = data_price_crop.columns.get_loc(data_price_crop.columns[data_price_crop.columns.str.startswith('Y1991')][0])
    data_us_crop = data_price_crop[data_price_crop['Area Code (M49)']==840]

    for climate_data in climate_names:
        yr_end = dict_year[climate_data]
        for v_model in model_names:
            fname = f'{v_model}_{climate_data}_hist_default_noirr_yield_{crop_mod[0:3]}_annual_anomaly_1981_{yr_end}.nc'
            ds     = xr.open_dataset(f'{dir_mod_adj}/{fname}')
            ds_org = xr.open_dataset(f'{dir_mod_org}/{fname}')
            data_yield         = ds['yield'].values
            data_yield_ana     = ds['yield_ana'].values
            data_yield_ana_org = ds_org['yield_ana'].values
            data_yield_org     = ds_org['yield'].values
    
            data_damage = np.minimum(data_yield-data_yield_org, 0)
            data_area_years = np.broadcast_to(crop_area, data_yield.shape)
            data_damage_area = - data_damage * data_area_years

            data_damage_area = xr.DataArray(data_damage_area, dims=['time', 'lat', 'lon'],\
                                            coords={'time': ds['time'], 'lat': ds['lat'], 'lon': ds['lon']})   
            us_boundary = np.broadcast_to(data_country, data_damage_area.shape)
            data_damage_area = xr.where(us_boundary != 840, np.nan, data_damage_area)
            
            # to csv
            data_damage_area_csv = xarray2dataframe(data_damage_area, v_model)
            if v_model == model_names[0]:
                data_damage_area_aggregated = data_damage_area_csv.copy()
            else:
                data_damage_area_aggregated = data_damage_area_aggregated.merge(data_damage_area_csv, on=['time', 'lat', 'lon'], how='left')
        # print(data_damage_area_aggregated)
        
        #==== calculate loss USD
        for year in range(yr_start,yr_end+1):
            Yyear = f'Y{year}'
            if Yyear in data_us_crop.columns:
                us_price = data_us_crop[Yyear].to_numpy()[0]
            else:
                us_price = np.nan
            # print(us_price)
            selected_rows = data_damage_area_aggregated['time'] == year
            data_damage_area_aggregated.loc[selected_rows, model_names] *= us_price * 0.5

        #===== save loss USD
        data_compare = data_obs.merge(data_damage_area_aggregated, on=['time', 'lat', 'lon'], how='outer')
        f_name = f'{crop_mod}_usd_noirr_{climate_data}'

        # add fips
        data_fips = data_compare[['lat', 'lon']].drop_duplicates().copy()
        data_fips['FIPS'] = data_fips.apply(lambda row: find_fips_vectorized(row, county_info), axis=1)
        data_compare = pd.merge(data_compare, data_fips, on=['lat', 'lon'], how='left')
        data_compare['state_id'] = data_compare['FIPS'].str[0:2]
        # print(data_compare)        
        # save .csv
        data_compare.to_csv(f'{f_indemnity}/{f_name}_fips.csv',index=False)




    