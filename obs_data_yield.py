import pandas as pd
import numpy as np
import os


def my_read_csv(csv_filename, cols):
    df = pd.read_csv(csv_filename, usecols=cols, thousands=',',
                     dtype={'State ANSI':str, 'County ANSI':str}, low_memory=False)
    return df

def load_usda_data(data_dir, data_startname):
    files = os.listdir(data_dir)
    files_all = [f for f in files if f.startswith(data_startname)]

    df = pd.DataFrame()
    for f in files_all:
        print(f'file name: {f}')
        df_temp = my_read_csv(f'{data_dir}/{f}', cols)
        df_temp['FIPS'] = df_temp['State ANSI'].str.zfill(2) + df_temp['County ANSI'].str.zfill(3)   
        df = pd.concat([df,df_temp],axis=0)
    df['Value'] = df['Value'].replace(r'.*[a-zA-Z].*', np.nan, regex=True)
    df['Value'] = df['Value'].replace(r',', '', regex=True).astype(float)
    df.dropna(subset=['FIPS','Value'], inplace=True)
    return df

def custom_agg_mean(s):
    if s.name == 'Yield':
        return s.mean()
    else:
        return s.iloc[0]
    
def custom_agg_sum(s):
    if s.name == 'Areaharvested':
        return s.sum()
    else:
        return s.iloc[0]

def process_usda_data(crop_name):

    data_startname = f'{crop_name}-yield'
    df = load_usda_data(data_dir, data_startname)
    print(df.shape)
    df.rename(columns={'Value':'Yield'},inplace=True)
    df.rename(columns={'Data Item':'Data Item-Yield'},inplace=True)
    df.rename(columns={'CV (%)':'CV-Yield (%)'},inplace=True)
    df = df.groupby(['FIPS', 'Year']).agg(custom_agg_mean).reset_index()
    print(df.shape)
    df_yield = df[['Year','State','FIPS','Data Item-Yield','Yield','CV-Yield (%)']].copy()   
    
    #======== read original data ========
    data_startname = f'{crop_name}-harvestarea'
    df = load_usda_data(data_dir, data_startname)
    print(df.shape)
    df.rename(columns={'Value':'Areaharvested'},inplace=True)
    df.rename(columns={'Data Item':'Data Item-Areaharvested'},inplace=True)
    df.rename(columns={'CV (%)':'CV-Areaharvested (%)'},inplace=True)
    df = df.groupby(['FIPS', 'Year']).agg(custom_agg_sum).reset_index()
    print(df.shape)
    df_harvestarea = df[['Year','State','FIPS','Data Item-Areaharvested','Areaharvested','CV-Areaharvested (%)']].copy()  

    #======== merge data ========   
    same_columns = ['Year','FIPS','State']
    df = pd.merge(df_yield,df_harvestarea,how='outer',on=same_columns)    
    print(df.shape)
    
    df_both = df[(~df['Yield'].isnull()) & (~df['Areaharvested'].isnull())]
    print(df_both.shape) 
    print(df_both.shape[0]/df.shape[0])
    print('\n')

    #======== save data ========
    save_flag = 1
    if save_flag:        
        start_year = df_both['Year'].min()
        end_year = df_both['Year'].max()
        df_both.to_csv(f'{output_dir}/{crop_name}_both_{start_year}-{end_year}.csv',index=False)

        start_year = df['Year'].min()
        end_year = df['Year'].max()
        df.to_csv(f'{output_dir}/{crop_name}_single_{start_year}-{end_year}.csv',index=False)

if __name__ == '__main__':

    data_dir = '../../data/USDA-YIELD/org_data/'
    output_dir = '../output/usda_yield_data/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cols = ['Year','State', 'State ANSI', 'County','County ANSI','Data Item','Value','CV (%)']
    
    name_crop = ['corn','soybeans','wheat']
    for ic in range(len(name_crop)): 
        crop_name = name_crop[ic]
        print(f'crop name: {crop_name}')
        process_usda_data(crop_name)
        print(f'Yield data for: {crop_name} saved! \n')
