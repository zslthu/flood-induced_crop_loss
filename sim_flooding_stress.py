import os
import glob
import re
import time
from multiprocessing import Pool
import xarray as xr
import numpy as np
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings('ignore')


def read_soil_parameter():
    ####### ks
    dir_ks  = '../data/k_s.nc'
    ds_ks   = xr.open_dataset(dir_ks)
    data_ds = np.nanmean(ds_ks['K_S'].values, axis=0)
    data_normlz_ks = np.nanmean(data_ds) / data_ds

    ####### water holding capacity
    dir_wr  = f'../data/Water_Holding_Capacities/wr_alldata.nc'
    ds_wr   = xr.open_dataset(dir_wr)
    data_wr = ds_wr['wrroot'].values
    data_wr[data_wr==0] = np.nan
    data_normlz_wr = np.nanmean(data_wr) / data_wr
    data_normlz_wr = np.flip(data_normlz_wr,axis=0)

    data_normlz = data_normlz_ks * data_normlz_wr

    return data_normlz    


def validate_files(v_model, v_crop, v_clim, name_var, dir_crop_yield):
    file_check = 0
    file_crop_yield_info = {}
    for v_var in name_var:
        file_pattern = f"{dir_crop_yield}/{v_model}/{v_model}.{v_crop}/{v_clim}/*{v_clim}*{name_sce[0]}*{name_irr[0]}*{v_var}*"
        # ##==== for isimip
        # file_pattern = f'{dir_crop_yield}/{v_model}/{v_clim}/{v_model}_{v_clim}_w5e5_{ssp}_2015soc_default_{v_var}_global_annual-gs_2015_2100.nc'
        file_list = glob.glob(file_pattern)
        if file_list:
            file_check += 1
            file_crop_yield_info[v_var] = file_list[0]
    return file_crop_yield_info, file_check


def read_data_info(file_crop_yield_info):
    data_info = {}
    for v_var, file_path in file_crop_yield_info.items():
        match = re.search(r'(\d{4})_(\d{4})\.nc4', file_path)
        # ##==== for isimip
        # match = re.search(r'(\d{4})_(\d{4})\.nc', file_path)
        if match:
            y_start_c, y_end_c = map(int, match.groups())
            data_info['time']= (y_start_c, y_end_c)
            data_info[v_var] = xr.open_dataset(file_path, decode_times=False)

    return data_info

def process_each_year(args):
    data_info, name_var, def_n, iy = args

    y_start_c = data_info['time'][0]
    year = y_start_c + iy
    fldfrc_file = f'{dir_flooding}/flooding_{year}_5day_0.5deg.nc'
    data_fldfrc = xr.open_dataset(fldfrc_file)
    # fldfrc = data_fldfrc['fldfrc_cont'].values  
    flddph = data_fldfrc['flddph_cont'].values

    crop_yield_year, crop_plant_year, crop_maty_year = \
        [data_info[v_var][v_var][iy, :, :] for v_var in name_var]

    plant_time = np.broadcast_to(crop_plant_year, flddph.shape)
    maty_time = np.broadcast_to(crop_maty_year, flddph.shape)
    growDay = crop_maty_year - crop_plant_year
    growDay_time = np.broadcast_to(growDay, flddph.shape)
    day_tmp = np.arange(1,flddph.shape[0]+1)[:, None, None]
    mask1 = (day_tmp >= plant_time) & (day_tmp <= maty_time)
    mask2 = (day_tmp >= plant_time) | (day_tmp <= maty_time)
    mask = np.where(growDay_time > 0, mask1, mask2)          

    flddph_crop = np.where(mask, flddph, 0)
    flddph_cont_max = np.nanmax(flddph_crop, axis=0)

    def_fv = def_n * data_normlz
    fadj = flddph_cont_max / (1 + flddph_cont_max ** def_fv) ** (1/def_fv)

    yield_adj = crop_yield_year * (1 - fadj)
    yield_adj = np.where(yield_adj < 0, 0, yield_adj)
    yield_org = crop_yield_year

    return yield_adj, yield_org

def process_data(name_var, data_info, def_n):
    y_start_c, y_end_c = data_info['time']

    # Use multiprocessing to process each year
    with Pool(processes=48) as pool:
        args_list = [(data_info, name_var, def_n, iy) for iy in range(y_end_c - y_start_c + 1)]
        results = pool.map(process_each_year, args_list)

    yield_adj, yield_org = zip(*results)
    yield_adj = np.stack(yield_adj, axis=0)
    yield_org = np.stack(yield_org, axis=0)
    
    adj_mean = np.nanmean(yield_org, axis=0) / np.nanmean(yield_adj, axis=0)
    yield_adj = yield_adj * adj_mean
    
    return yield_adj, yield_org

def save_results(yield_adj, yield_org, v_model, v_clim, name_var, y_start_c, y_end_c):
    grid = 0.5
    
    f_dir = f'{dir_output}/{v_model}/'
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)

    f_name = f'{v_model}_{v_clim}_hist_{name_sce[0]}_{name_irr[0]}_{name_var[0]}_annual_{y_start_c}_{y_end_c}.nc4'
    # ##=== for isimip
    # f_name = f'{v_model}_{v_clim}_w5e5_{ssp}_2015soc_default_{name_var[0]}_global_annual-gs_2015_2100.nc' 

    ds = xr.Dataset({       name_var[0]: (['time','lat', 'lon'], yield_adj),
                            'yield_org': (['time','lat', 'lon'], yield_org),
                        },
                    coords= {'lon': np.arange(-180+grid/2, 180, grid),
                             'lat': np.arange(90-grid/2,   -90, -grid),
                            'time': np.arange(y_start_c, y_end_c + 1),
                            })

    ds.to_netcdf(f_dir + f_name)
    print(f'Saved: {f_name} \n')


if __name__ == "__main__":

    ## simulation information
    name_model     = ['cgms-wofost','lpj-guess','clm-crop','lpjml','epic-iiasa','gepic','orchidee-crop','papsim','pdssat','pegasus']
    name_climate   = ['wfdei.gpcc','agmerra']
    name_sce       = ['default']
    name_irr       = ['noirr']
    name_var_org   = ['yield', 'plant-day','maty-day']
    name_crop      = ['maize','soy','wheat']

    ## Define directories
    dir_crop_yield  = '/tera05/zhangsl/GGCMI_phase1_unzipped'
    dir_flooding  = '/tera05/zhangsl/cama/hist/flooding'
    dir_output  = f'../output/AgMIP_adj/'
    os.makedirs(dir_output, exist_ok=True)

    # ##==== for isimip
    # ## simulation information
    # name_model     = ['crover', 'epic-iiasa', 'ldndc', 'lpj-guess', 'lpjml', 'pdssat', 'pepic', 'promet', 'simplace-lintul5'] 
    # name_climate   = ['gfdl-esm4','ukesm1-0-ll','mri-esm2-0','mpi-esm1-2-hr','ipsl-cm6a-lr']
    # name_sce       = ['default']
    # name_irr       = ['noirr']
    # name_var_org   = ['yield', 'plantday','matyday']
    # name_crop      = ['maize','soy','wwh','swh']

    # ssp = 'ssp585'

    # ## Define directories
    # dir_crop_yield  = f'/tera07/zhangsl/lianghb21/ISMIP/ISMIP_3b/crop_{ssp}'
    # dir_output  = f'../output/ISIMIP_adj/{ssp}'

    data_normlz = read_soil_parameter()

    crop_n = [0.64, 0.70, 0.60]
    for ic in range(1):
        v_crop = name_crop[ic]
        name_var = [f'{v_var}_{v_crop[0:3]}' for v_var in name_var_org]
        # ##==== for isimip
        # name_var = [f'{v_var}-{v_crop[0:3]}-noirr' for v_var in name_var_org]

        for v_clim in name_climate[0:1]:
            # ##==== for isimip
            # dir_flooding  = f'/tera05/zhangsl/cama/ismip_new/{ssp}/ensemble/{v_clim}/flooding'

            for v_model in name_model:            

                def_n  = crop_n[ic]
                file_crop_yield_info, file_check = validate_files(v_model, v_crop, v_clim, name_var, dir_crop_yield)

                if file_check == len(name_var):
                    print(f'model = {v_model} / climate = {v_clim} / crop = {v_crop}...')                

                    data_info = read_data_info(file_crop_yield_info)

                    yield_adj, yield_org = process_data(name_var, data_info, def_n)

                    save_results(yield_adj, yield_org, v_model, v_clim, name_var, data_info['time'][0], data_info['time'][1])
                        
    