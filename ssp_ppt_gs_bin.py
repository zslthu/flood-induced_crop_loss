import xarray as xr
import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
from multiprocessing import Pool
from multiprocessing import Array
import ctypes
import time

def cal_clim_bin(P):

    P['Prec126_percentile']=(stats.rankdata(P['Prec126'], method='average')*2-1)/(P['Prec126'].shape[0]*2)
    P['Prec585_percentile']=(stats.rankdata(P['Prec585'], method='average')*2-1)/(P['Prec585'].shape[0]*2)

    P_bin = P.copy()
    v_mean = np.mean(np.concatenate([P_bin['Prec126'], P_bin['Prec585']]))
    v_std = np.std(np.concatenate([P_bin['Prec126'], P_bin['Prec585']]))

    prec_bin_sigma = [v_mean + i * v_std for i in np.arange(-3.5,3.6,0.5)]
    prec_bin_rank = np.arange(0,1.0001,0.05)

    # ssp585
    bin_means1, bin_edges1, binnumber1 = stats.binned_statistic(P_bin['Prec585_percentile'], P_bin['Prec585_percentile'], 'mean', bins=prec_bin_rank)
    bin_means2, bin_edges2, binnumber2 = stats.binned_statistic(P_bin['Prec585'], P_bin['Prec585'], 'mean', bins=prec_bin_sigma)
    P_bin['Prec585_rank_bin'] =  binnumber1
    P_bin['Prec585_sigma_bin'] = binnumber2
    P_bin['Prec585_to_sd'] = (P_bin['Prec585'] - v_mean)/v_std

    # ssp126
    bin_means1, bin_edges1, binnumber1 = stats.binned_statistic(P_bin['Prec126_percentile'], P_bin['Prec126_percentile'], 'mean', bins=prec_bin_rank)
    bin_means2, bin_edges2, binnumber2 = stats.binned_statistic(P_bin['Prec126'], P_bin['Prec126'], 'mean', bins=prec_bin_sigma)
    P_bin['Prec126_rank_bin'] =  binnumber1
    P_bin['Prec126_sigma_bin'] = binnumber2
    P_bin['Prec126_to_sd'] = (P_bin['Prec126'] - v_mean)/v_std

    # print(P_bin[['Prec126','Prec126_sigma_bin']], P_bin[['Prec585','Prec585_sigma_bin']])
    # print(P_bin)
    return P_bin['Prec126_sigma_bin'].values, P_bin['Prec585_sigma_bin'].values

def process_data_point(args):
    ii, jj = args
    data_ppt_times126 = data_ppt_ssp126[:, ii, jj]
    data_ppt_times585 = data_ppt_ssp585[:, ii, jj]
    if (np.std(data_ppt_times126) != 0) and (np.std(data_ppt_times585) != 0):
        P = pd.DataFrame({'Year': np.arange(ys, ye + 1), 'Prec126': data_ppt_times126, 'Prec585': data_ppt_times585})
        # calculate sigma bin
        P_sigma_bin126, P_sigma_bin585 = cal_clim_bin(P)  
        return ii, jj, P_sigma_bin126, P_sigma_bin585
    else:
        return ii, jj, None, None


def parallel_process_data_sigma_bin():
    data_sigma_bin126 = np.zeros(data_ppt_ssp126.shape) * np.nan
    data_sigma_bin585 = np.zeros(data_ppt_ssp585.shape) * np.nan

    indices = list(product(range(360), range(720)))
    args_list = [(ii, jj) for ii, jj in indices]

    with Pool() as pool:
        results = pool.map(process_data_point, args_list)
    
    for result in results:
        if result is not None:
            ii, jj, P_sigma_bin126, P_sigma_bin585 = result
            if P_sigma_bin126 is not None:
                data_sigma_bin126[:, ii, jj] = P_sigma_bin126
            if P_sigma_bin585 is not None:
                data_sigma_bin585[:, ii, jj] = P_sigma_bin585

    return data_sigma_bin126, data_sigma_bin585

#=========== GGCMI data information
name_model     = ['crover', 'epic-iiasa', 'ldndc', 'lpj-guess', 'lpjml', 'pdssat', 'pepic', 'promet', 'simplace-lintul5'] 
name_clim      = ['gfdl-esm4','ukesm1-0-ll','mri-esm2-0','mpi-esm1-2-hr','ipsl-cm6a-lr']
name_sce       = ['default']
name_irr       = ['noirr']
name_var_org   = ['yield', 'plantday','matyday']
name_crop      = ['mai','soy','whe']

ys = 2015
ye = 2100
ppt_ys = [2015, 2021, 2031, 2041, 2051, 2061, 2071, 2081, 2091]
ppt_ye = [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
grid  = 0.5

start_time = time.time()

dir_ppt     = f'/tera07/zhangsl/lianghb21/ISMIP/ISMIP_3b/climate/ppt_growing_season/'
dir_ppt_out = f'/tera07/zhangsl/lianghb21/ISMIP/ISMIP_3b/climate/ppt_growing_season/'
for v_clim in name_clim:
    for v_crop in name_crop:
        for v_model in name_model:
            print(f'Processing: {v_clim}_{v_model}_{v_crop}')
            # for ssp in name_ssp:
            f_name = f'ppt_gs_ssp585_{v_clim}_{v_model}_{v_crop}_2015_2100.nc'
            data = xr.open_dataset(dir_ppt + f_name, decode_times=False)
            # print(data)                
            data_ppt_ssp585 = data['ppt_crop'].values

            f_name = f'ppt_gs_ssp126_{v_clim}_{v_model}_{v_crop}_2015_2100.nc'
            data = xr.open_dataset(dir_ppt + f_name, decode_times=False)
            # print(data)                
            data_ppt_ssp126 = data['ppt_crop'].values                

            #==========parallel_process_data_sigma_bin
            data_sigma_bin126, data_sigma_bin585 = parallel_process_data_sigma_bin()

            #==========save nc
            f_name = f'unify_sigma_bin_ssp126_{v_clim}_{v_model}_{v_crop}_2015_2100.nc' 
            ds = xr.Dataset({   'sigma_bin': (['time','lat', 'lon'], data_sigma_bin126)},
                        coords= {    'lon': np.arange(-180+grid/2, 180, grid),
                                     'lat': np.arange(90-grid/2,   -90, -grid),
                                    'time': np.arange(ys, ye + 1),
                                    })
            ds.to_netcdf(dir_ppt_out + f_name)
            print(f'Saved: {f_name} \n')  

            f_name = f'unify_sigma_bin_ssp585_{v_clim}_{v_model}_{v_crop}_2015_2100.nc' 
            ds = xr.Dataset({   'sigma_bin': (['time','lat', 'lon'], data_sigma_bin585)},
                        coords= {    'lon': np.arange(-180+grid/2, 180, grid),
                                     'lat': np.arange(90-grid/2,   -90, -grid),
                                    'time': np.arange(ys, ye + 1),
                                    })
            ds.to_netcdf(dir_ppt_out + f_name)
            print(f'Saved: {f_name} \n')   

                
end_time = time.time()
print(f'Elapsed time: {end_time - start_time} seconds')



 
