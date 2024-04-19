import os
import numpy as np
import pandas as pd
import cartopy.io.shapereader as shpreader
from affine import Affine

# source: Li, Y., Guan, K., Schnitkey, G. D., DeLucia, E., & Peng, B. (2019). 
# Excessive rainfall leads to maize yield loss of a comparable magnitude to extreme drought in the United States.
# Global Change Biology, 25(7), 2325–2337. https://doi.org/10.1111/gcb.14628
def get_county_latlon(rerun=True):
    
    save_dir = '../output'
    os.makedirs(save_dir, exist_ok=True)

    if rerun:
        county_shapes = shpreader.Reader(dir_county_shapes)
        state_shapes = shpreader.Reader(dir_state_shapes)
        county_rec = list(county_shapes.records())
        county_geo = list(county_shapes.geometries())
        
        fips = [county_rec[f].attributes['FIPS'] for f in range(len(county_rec))]
        lon = [(county_geo[i].bounds[0] + county_geo[i].bounds[2])/2 for i in range(len(county_rec))]
        lat = [(county_geo[i].bounds[1] + county_geo[i].bounds[3])/2 for i in range(len(county_rec))]
        area = [county_rec[f].attributes['AREA'] for f in range(len(county_rec))]
        
        d = {'FIPS': fips,
         'lat': lat,
         'lon':lon,
         'county_area':area}

        df = pd.DataFrame(d, columns=['FIPS', 'lat', 'lon', 'county_area'])

        # Define Affine of 0.5 degree
        a = Affine(0.5,0,-180,0,-0.5,90)
        # get col and row number
        df['col'], df['row'] = ~a * (df['lon'], df['lat']) 
        # need to floor to get integer col and row

        df['col'] = df['col'].apply(np.floor).astype(int)
        df['row'] = df['row'].apply(np.floor).astype(int)


        df.to_csv(f'{save_dir}/county_latlon.csv', index=False)        
        print('Extracting lat lon for each county done. File county_latlon.csv saved')

    else:

        df = pd.read_csv(f'{save_dir}/county_latlon.csv',dtype={'FIPS':str})

    return df

if __name__=='__main__':
    dir_county_shapes = '../data/counties_contiguous/counties_contiguous.shp'
    dir_state_shapes  = '../data/states_contiguous/states_contiguous.shp'

    save_dir = '../output'
    os.makedirs(save_dir, exist_ok=True)

    get_county_latlon(rerun=True)