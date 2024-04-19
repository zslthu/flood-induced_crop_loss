import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import multiprocessing

######################################## function
def read_country_code():
    #====== Create a meshgrid for the new grids
    grid = 5/60
    nx = int(360/grid)
    ny = int(180/grid)
    lon_intp = np.arange(-180+grid/2, 180, grid)
    lat_intp = np.arange(90-grid/2,   -90, -grid)

    #====== read country code
    data_country = xr.open_dataset(dir_country)['Band1'].values
    data_country = np.flip(data_country, axis=0)
    code_country = np.unique(data_country[~np.isnan(data_country)]).astype(int)

    data_country_name = pd.read_csv('../data/crop_price/M49_name.csv')
    data_loss_usd  = pd.DataFrame({'M49': code_country})
    data_loss_usd  = pd.merge(data_loss_usd, data_country_name, how='inner', on='M49')
    data_loss_production = data_loss_usd.copy()
    num_country  = len(data_loss_usd)
    
    return data_country, data_country_name, data_loss_usd, data_loss_production, num_country, lat_intp, lon_intp

def read_price(v_crop):
    dir_price  = f'../data/crop_price/crop_price_fillna_{v_crop}.csv'
    data_price = pd.read_csv(dir_price)
    idx_y = data_price.columns.get_loc(data_price.columns[data_price.columns.str.startswith('Y1991')][0])
    idx_end = data_price.columns.get_loc(data_price.columns[data_price.columns.str.startswith('Y2015')][0])
    data_price['mean'] = data_price.iloc[:,idx_y:idx_end].mean(axis=1)
    data_price['max']  = data_price.iloc[:,idx_y:idx_end].max(axis=1)
    data_price['min']  = data_price.iloc[:,idx_y:idx_end].min(axis=1)
    data_price['2015'] = data_price['Y2015']
    # print(data_price)

    return data_price

def cal_damage_area(start_string):
    # print(f'Processing {v_crop}-{ssp}{org}-{start_string}...')
    fname = f'{start_string}_{ssp}_default_noirr_yield_{v_crop[0:3]}_annual_anomaly_2015_2100.nc'
    dname = f'{dir_mod_out}/{ssp}/{fname}'
    if not os.path.exists(dname):
        print(f'File not exists: {dname}')
        return np.nan
    ds = xr.open_dataset(dname)
    ds_new = ds.interp(lat=lat_intp, lon=lon_intp, method='nearest')
    data_yield_ana    = ds_new['yield_ana'].values
    data_yield_trend  = ds_new['yield_trend'].values

    # read extreme wet and dry year
    v_model = start_string.split('_')[0]
    v_clim = start_string.split('_')[1]
    f_name = f'unify_sigma_bin_{ssp}_{v_clim}_{v_model}_{v_crop[0:3]}_2015_2100.nc' 
    ds_bin = xr.open_dataset(dir_ppt_out + f_name)
    ds_bin_new = ds_bin.interp(lat=lat_intp, lon=lon_intp, method='nearest')
    # print(ds_bin)
    data_p_bin = ds_bin_new['sigma_bin'].values

    data_yield_ana_damage = data_yield_ana
    if dw == 'wet':
        # extreme wet
        data_yield_ana_damage[data_p_bin<12] = np.nan
    else:
        # extreme dry
        data_yield_ana_damage[data_p_bin>4] = np.nan

    data_area_years = np.broadcast_to(crop_area, data_yield_ana.shape)
    data_damage_area = - np.minimum(data_yield_ana_damage, 0) * data_area_years

    return data_damage_area

def cal_loss_each_country(i,data_damage_area,data_price,price_level):
    c_country = data_loss_usd['M49'].to_numpy()[i]
    # print(c_country)
    idx_country_area  = data_country == c_country
    # print(idx_country_area)
    idx_country_price = data_price['Area Code (M49)'] == c_country
    if np.sum(idx_country_price) != 0:
        # continue
        for year in range(yr_start,yr_end+1):
            data_damage_area_year = data_damage_area[year-yr_start,:,:]                
            area_country = np.nansum(data_damage_area_year[idx_country_area])
            if price_level in data_price.columns:
                price = data_price[price_level].to_numpy()[idx_country_price]
            else:
                price = np.nan
            loss_country = area_country * price * 0.5

            Yyear = f'Y{year}'
            if Yyear not in data_loss_usd.columns:
                data_loss_usd[Yyear] = np.nan
                data_loss_production[Yyear] = np.nan
            data_loss_usd.loc[data_loss_usd['M49'] == c_country, Yyear]  = loss_country
            data_loss_production.loc[data_loss_production['M49'] == c_country, Yyear] = area_country
    
    return data_loss_usd, data_loss_production


def process_each_model(start_string,price_level):
    data_damage_area = cal_damage_area(start_string)

    if np.isnan(data_damage_area).all():
        print(f'No data: {start_string}')
    else:
        print(f'Processing {dw}-{v_crop}-{ssp}-{start_string}...')
        for i in range(num_country):
            data_loss_usd, data_loss_production = cal_loss_each_country(i,data_damage_area,data_price,price_level)

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

        data_loss_usd.to_csv(f'{f_loss}/compare_ew_ed/{dw}_loss_usd_{v_crop}_{start_string}_{ssp}.csv', index=False)
        data_loss_production.to_csv(f'{f_loss}/compare_ew_ed/{dw}_loss_production_{v_crop}_{start_string}_{ssp}.csv', index=False)
            
######################################## function

if __name__ == '__main__':

    #====== model information
    name_crop_mod = ['maize','soy','wheat']
    name_model    = ['crover','epic-iiasa', 'ldndc', 'lpj-guess', 'lpjml', 'pdssat', 'pepic', 'promet', 'simplace-lintul5']
    name_climate  = ['gfdl-esm4','ukesm1-0-ll','mri-esm2-0','mpi-esm1-2-hr','ipsl-cm6a-lr']
    name_ssp      = ['ssp126','ssp585']
    name_dw       = ['wet','dry']

    #====== file dir ======
    #====== dir for crop price and country code
    dir_country = '../data/crop_price/M49_05mn.nc'

    #====== dir for crop area
    dir_crop  = '../data/crop_area/crop_area_05mn.nc'
    
    #====== dir for extreme wet and dry year
    dir_ppt_out   = f'/tera07/zhangsl/lianghb21/ISMIP/ISMIP_3b/climate/ppt_growing_season/'

    #====== set up ======
    yr_start = 2015
    yr_end   = 2100

    #====== dir for crop yield
    fn = 'org'
    dir_mod_out = f'../output/anomaly_mod/isimip_{fn}'
    f_loss_save = f'../output/loss_glb_ssp/ssp_{fn}' 

    #========================================================
    # read country code
    data_country, data_country_name, data_loss_usd, data_loss_production, num_country, lat_intp, lon_intp = read_country_code()
    # read crop area
    data_area = xr.open_dataset(dir_crop)

    start_string_all = [(model+'_'+climate) for model in name_model for climate in name_climate]
    
    #======================== main ==========================
    for ssp in name_ssp[0:1]:
        f_loss = f'{f_loss_save}/{ssp}'
        os.makedirs(f'{f_loss}/compare_ew_ed', exist_ok=True)

        for ic in range(1):
            v_crop    = name_crop_mod[ic]
            data_loss_usd['Item']  = v_crop
            data_loss_production['Item'] = v_crop

            #====== [var A] read crop area for each crop
            crop_area = data_area[v_crop].values
            # print(crop_area.shape)

            #====== [var P] read crop price for each crop
            data_price  = read_price(v_crop)            
            price_level = 'mean'

            for dw in name_dw:
                pool = multiprocessing.Pool(processes=10)
                pool.starmap(process_each_model, [(start_string,price_level) for start_string in start_string_all])
                pool.close()
                pool.join()
