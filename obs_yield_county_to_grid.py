import numpy as np
import pandas as pd
import xarray as xr
import os
from get_county_latlon import get_county_latlon


def get_grid_yield_area(crop):

    data_combined = pd.read_csv(csv_filename,dtype={'FIPS':str})
    data_combined = data_combined.merge(county_info, on='FIPS')

    data_combined = data_combined.fillna(0)
    data_combined['indemnity_Excess Moisture/Precip/Rain'] = \
        data_combined['indemnity_Excess Moisture/Precip/Rain'] + data_combined['indemnity_Flood']


    array_indemnity = np.zeros([ny,nx,len_year])
    array_area  = np.zeros([ny,nx,len_year])
    array_yield = np.zeros([ny,nx,len_year])
    array_yield_ana = np.zeros([ny,nx,len_year])
    array_yield_ana_to_yield = np.zeros([ny,nx,len_year])
    array_Prec_sigma_bin = np.zeros([ny,nx,len_year])
    for y in range(start_year, end_year+1):
        temp = data_combined[data_combined['Year']==y]
        temp = temp.dropna()
        # print(temp)

        for n in range(temp.shape[0]):
            r = temp.iloc[n,:]['row']
            c = temp.iloc[n,:]['col']

            array_indemnity[r,c,y-start_year] = array_indemnity[r,c,y-start_year] + \
                temp.iloc[n,:]['Area']*temp.iloc[n,:]['indemnity_Excess Moisture/Precip/Rain']

            array_area[r,c,y-start_year]  = array_area[r,c,y-start_year] + temp.iloc[n,:]['Area']
            array_yield[r,c,y-start_year] = array_yield[r,c,y-start_year] + temp.iloc[n,:]['Area']*temp.iloc[n,:]['Yield']

            array_yield_ana[r,c,y-start_year] = array_yield_ana[r,c,y-start_year] + \
                temp.iloc[n,:]['Area']*temp.iloc[n,:]['Yield_ana']
            array_yield_ana_to_yield[r,c,y-start_year] = array_yield_ana_to_yield[r,c,y-start_year] + \
                temp.iloc[n,:]['Area']*temp.iloc[n,:]['Yield_ana_to_yield']

            array_Prec_sigma_bin[r,c,y-start_year] = array_Prec_sigma_bin[r,c,y-start_year] + \
                temp.iloc[n,:]['Area'] *temp.iloc[n,:]['Prec_sigma_bin']

    array_area[array_area==0] = np.nan

    array_indemnity = array_indemnity/array_area

    array_yield     = array_yield/array_area
    array_yield_ana = array_yield_ana/array_area
    array_yield_ana_to_yield = array_yield_ana_to_yield/array_area

    array_Prec_sigma_bin = array_Prec_sigma_bin/array_area
    array_Prec_sigma_bin = np.round(array_Prec_sigma_bin)


    to_nc = 1
    if to_nc:
        lon = np.arange(-180+grid/2, 180,  grid)
        lat = np.arange(90-grid/2,   -90, -grid)
        ds  = xr.Dataset({         'area': (['time','lat', 'lon'], np.transpose(array_area, (2, 0, 1))),
                                  'yield': (['time','lat', 'lon'], np.transpose(array_yield, (2, 0, 1))),
                              'yield_ana': (['time','lat', 'lon'], np.transpose(array_yield_ana, (2, 0, 1))),
                     'yield_ana_to_yield': (['time','lat', 'lon'], np.transpose(array_yield_ana_to_yield, (2, 0, 1))),
                         'Prec_sigma_bin': (['time','lat', 'lon'], np.transpose(array_Prec_sigma_bin, (2, 0, 1))),
                              'indemnity': (['time','lat', 'lon'], np.transpose(array_indemnity, (2, 0, 1))),
                                },
                        coords={'lon': lon,
                                'lat': lat,
                                'time': np.arange(start_year,end_year+1,1)
                                    })
        # save result
        output_name = f'{crop}_grid_climbin_yield_{start_year}_{end_year}_05deg.nc'
        ds.to_netcdf(f'{output_dir}/{output_name}')
        print(f'File {output_name} saved!')

        to_csv = 0
        if to_csv:
            df = ds[['yield_ana_to_yield','area','Prec_sigma_bin','yield','yield_ana']].to_dataframe().reset_index()
            df = df.dropna()
            output_name = f'{crop}_obs_yield_clim_anom_{start_year}_{end_year}_05deg.csv'
            df.to_csv(f'{output_dir}/{output_name}', index=False)


#========================================================
# Load USDA-YIELD data
usda_dir    = '../output/obs_data'
output_dir  = '../output/obs_data'

# Set up grid and time
start_year = 1981
end_year   = 2010
len_year   = end_year - start_year + 1

grid       = 0.5
nx         = int(360/grid)
ny         = int(180/grid)

# Load county information
county_info = get_county_latlon(rerun=False)

# main calculaiton
name_crop = ['corn','soybeans','wheat']
for ic in range(3):
    crop = name_crop[ic]

    csv_filename = f'{usda_dir}/{crop}_climbin_yield_anomaly_loss_linear_indemnity.csv'
    get_grid_yield_area(crop)

    


