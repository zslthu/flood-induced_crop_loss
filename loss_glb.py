import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import multiprocessing


def read_country_code():
    # # Create a meshgrid for the new grids
    grid = 5/60
    nx = int(360/grid)
    ny = int(180/grid)
    lon_intp = np.arange(-180+grid/2, 180, grid)
    lat_intp = np.arange(90-grid/2,   -90, -grid)

    # print(num_country)
    dir_country  = '../data/crop_price/M49_05mn.nc'
    data_country = xr.open_dataset(dir_country)['Band1'].values
    data_country = np.flip(data_country, axis=0)
    code_country = np.unique(data_country[~np.isnan(data_country)]).astype(int)

    data_country_name = pd.read_csv('../data/crop_price/M49_name.csv')
    data_loss_usd  = pd.DataFrame({'M49': code_country})
    data_loss_usd  = pd.merge(data_loss_usd, data_country_name, how='inner', on='M49')
    data_loss_production = data_loss_usd.copy()
    num_country    = len(data_loss_usd)

    return data_country, data_country_name, data_loss_usd, data_loss_production, num_country, lat_intp, lon_intp

######################################## function
def cal_damage_area(v_model):
    print(f'Processing {v_crop}-{v_model}...')
    fname = f'{v_model}_{climate_data}_hist_default_noirr_yield_{v_crop[0:3]}_annual_anomaly_1981_{dict_year[climate_data]}.nc'
    ds     = xr.open_dataset(f'{dir_mod_adj}/{fname}')
    ds_org = xr.open_dataset(f'{dir_mod_org}/{fname}')
    ds_new = ds.interp(lat=lat_intp, lon=lon_intp, method='nearest')
    ds_org_new = ds_org.interp(lat=lat_intp, lon=lon_intp, method='nearest')
    data_yield        = ds_new['yield'].values
    data_yield_ana    = ds_new['yield_ana'].values
    data_yield_org    = ds_org_new['yield'].values
    data_damage = np.minimum(data_yield-data_yield_org, 0)
    
    data_area_years  = np.broadcast_to(crop_area, data_yield.shape)
    data_damage_area = - data_damage * data_area_years 

    return data_damage_area

def cal_loss_each_country(i,data_damage_area,data_price):
    c_country = data_loss_usd['M49'].to_numpy()[i]
    # print(c_country)
    idx_country_area  = data_country == c_country 
    idx_country_price = data_price['Area Code (M49)'] == c_country
    if np.sum(idx_country_price) != 0:
        # continue
        for year in range(yr_start_price,yr_end+1):
            data_damage_area_year = data_damage_area[year-yr_start,:,:]                
            area_country = np.nansum(data_damage_area_year[idx_country_area])
            Yyear = f'Y{year}'
            if Yyear in data_price.columns:
                price = data_price[Yyear].to_numpy()[idx_country_price]
            else:
                price = np.nan
            loss_country = area_country * price * 0.5

            if Yyear not in data_loss_usd.columns:
                data_loss_usd[Yyear] = np.nan
                data_loss_production[Yyear] = np.nan
            data_loss_usd.loc[data_loss_usd['M49'] == c_country, Yyear] = loss_country
            data_loss_production.loc[data_loss_production['M49'] == c_country, Yyear] = area_country
    
    return data_loss_usd, data_loss_production

def process_each_model(v_model):
    data_damage_area = cal_damage_area(v_model)

    for i in range(num_country):
        data_loss_usd, data_loss_production = cal_loss_each_country(i,data_damage_area,data_price)

    new_row = data_loss_usd.iloc[0:1].copy()
    data_loss_usd = pd.concat([new_row, data_loss_usd], ignore_index=True)
    data_loss_usd.iloc[0,0] = 0
    data_loss_usd.iloc[0,1] = 'Global'
    idx_price = [i for i in range(len(data_loss_usd.columns)) if 'Y' in data_loss_usd.columns[i]]
    idx_price = idx_price[0]
    data_loss_usd.iloc[0,idx_price::] = np.nansum(data_loss_usd.iloc[1::, idx_price::].values, axis=0)

    new_row = data_loss_production.iloc[0:1].copy()
    data_loss_production = pd.concat([new_row, data_loss_production], ignore_index=True)
    data_loss_production.iloc[0,0] = 0
    data_loss_production.iloc[0,1] = 'Global'
    data_loss_production.iloc[0,idx_price::] = np.nansum(data_loss_production.iloc[1::, idx_price::].values, axis=0)

    data_loss_usd.to_csv(f'{f_loss}/loss_usd/loss_usd_{v_crop}_{v_model}_{climate_data}.csv', index=False)
    data_loss_production.to_csv(f'{f_loss}/loss_production/loss_production_{v_crop}_{v_model}_{climate_data}.csv', index=False)

######################################## function

if __name__ == '__main__':

    f_loss   = f'../output/loss_glb/'  
    os.makedirs(f'{f_loss}/loss_usd', exist_ok=True)
    os.makedirs(f'{f_loss}/loss_production', exist_ok=True)

    # # model information
    name_crop_obs = ['corn','soybeans','wheat']
    name_crop_mod = ['maize','soy','wheat']
    model_names   = ['cgms-wofost','lpj-guess','clm-crop','lpjml','epic-iiasa','gepic','orchidee-crop','pdssat','papsim','pegasus']
    climate_names = ['wfdei.gpcc','agmerra']

    # # dir for crop yield
    dir_mod_adj = '../output/anomaly_mod/agmip_adj/'
    dir_mod_org = '../output/anomaly_mod/agmip_org/'

    yr_start = 1981
    yr_end   = 2009
    dict_year = {'wfdei.gpcc':2009,'agmerra':2010}
    yr_start_price = 1991

    # # dir for crop area
    dir_crop  = '../data/crop_area/crop_area_05mn.nc'
    data_area = xr.open_dataset(dir_crop)

    data_country, data_country_name, data_loss_usd, data_loss_production, num_country, lat_intp, lon_intp = read_country_code()

    for climate_data in climate_names:
        for ic in range(3):
            v_crop    = name_crop_mod[ic]
            data_loss_usd['Item']  = v_crop
            data_loss_production['Item'] = v_crop

            # # [var A] read crop area for each crop
            crop_area = data_area[v_crop].values
            # print(crop_area.shape)

            # # [var P] read crop price for each crop
            dir_price  = f'../data/crop_price/crop_price_fillna_{v_crop}.csv'
            data_price = pd.read_csv(dir_price)
            # print(data_price)
            
            # process_each_model(model_names[0])
            pool = multiprocessing.Pool(processes=10)
            pool.map(process_each_model, model_names)
            pool.close()
            pool.join()


        

