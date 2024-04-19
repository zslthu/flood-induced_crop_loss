import numpy as np
import pandas as pd
import os

def del_space(s):
    if isinstance(s, str):
        return s.strip()
    else:
        return s 

def load_rma_loss(year):
    filename = data_dir + '/colsom/colsom_' + str(year) + '.txt'
    col_names = ['Commodity Year', 'Locations State Code', 'Location State Abbreviation',
                    'Location County Code', 'Location County Name', 'Commodity Code',
                    'Commodity Name', 'Insurance Plan Code', 'Insurance Plan Abbreviation',
                    'Coverage Category', 'Stage Code', 
                    'Damage Cause Code', 'Damage Cause Description',
                    'Month of Loss', 'Month of Loss Abbreviation',  'Year of Loss', 
                    'Policies Earning Premium','Policies Indemnified','Net Planted Quantity','Net Endorsed Acres',
                    'Liability','Total Premium Amount','Producer Paid Premium','Subsidy','State/Private Subsidy','Additional Subsidy',
                    'EFA Premium Discount','Net Determined Quantity','Indemnity Amount','Loss Ratio']
        
    d = pd.read_csv(filename, sep='|', header=None, names=col_names, index_col=False,
                    dtype={'Locations State Code': str, 'Location County Code': str,
                           'Commodity Code': str, 'Damage Cause Code': str},low_memory=False) 

    # delete extra space
    d['Commodity Name'] = d['Commodity Name'].apply(del_space)
    d['Damage Cause Description'] = d['Damage Cause Description'].apply(del_space)
    d['FIPS']=d['Locations State Code'] + d['Location County Code']
    if not d['FIPS'].str.len().eq(5).all():
        print(f'year = {year} data check: WRONG')

    save_col_names = ['FIPS','Commodity Year', 'Commodity Code','Commodity Name',
                    'Damage Cause Code', 'Damage Cause Description',
                    'Month of Loss', 'Month of Loss Abbreviation',  'Year of Loss', 
                    'Total Premium Amount','Indemnity Amount', 'Loss Ratio']
    d = d[save_col_names].copy()

    return d

def load_rma_sob(year):
    filename = data_dir + '/sobcov/sobcov_' + str(year) + '.txt'
    col_names = ['Commodity Year', 'Locations State Code', 'Location State Abbreviation',
                 'Location County Code', 'Location County Name', 'Commodity Code',
                 'Commodity Name', 'Insurance Plan Code', 'Insurance Plan Abbreviation',
                 'Coverage Category', 'Delivery Type', 'Coverage Level',
                 'Policies Sold Count', 'Policies Earning Premium Count', 'Policies Indemnified Count',
                 'Units Earning Premium Count', 'Units Indemnified Count', 'Quantity Type',
                 'Net Reported Quantity', 'Endorsed/Companion Acres', 'Liability Amount',
                 'Total Premium Amount', 'Subsidy Amount', 'State/Private Subsidy',
                 'Additional Subsidy ', 'EFA Premium Discount',
                 'Indemnity Amount', 'Loss Ratio']

    d = pd.read_csv(filename, sep='|', header=None, names=col_names, index_col=False,
                    dtype={'Locations State Code': str, 'Location County Code': str,
                           'Commodity Code': str},low_memory=False)
    d['FIPS']=d['Locations State Code'] + d['Location County Code']

    d['Commodity Name'] = d['Commodity Name'].apply(del_space)
    d['Quantity Type'] = d['Quantity Type'].apply(del_space)

    if not d['FIPS'].str.len().eq(5).all():
        print(f'year = {year} data check: WRONG')

    save_col_names = ['FIPS','Commodity Year', 'Commodity Code','Commodity Name',
                 'Total Premium Amount','Indemnity Amount', 'Loss Ratio']
    d = d[save_col_names].copy()    

    return d

def load_rma_loss_all(crop_name='corn',rerun=False):
    if rerun:
        frame = [load_rma_loss(i) for i in range(start_year, end_year+1)]
        data_loss = pd.concat(frame)
        data_loss = data_loss[(data_loss['Commodity Name']==crop_name.upper())]
        
        cause_dict = dict(zip(data_loss['Damage Cause Code'],data_loss['Damage Cause Description']))
        data_loss['Damage Cause Description'] = data_loss['Damage Cause Code'].map(cause_dict)
        
        data_loss.to_csv(f'{output_dir}/{crop_name}_loss_all.csv', index=False)
    else:
        data_loss = pd.read_csv(f'{output_dir}/{crop_name}_loss_all.csv',dtype={'FIPS':str},low_memory=False)

    return data_loss

def load_rma_sob_all(crop_name='corn',rerun=False):
    if rerun:
        frame = [load_rma_sob(i) for i in range(start_year, end_year+1)]
        data_sob = pd.concat(frame)
        data_sob = data_sob[(data_sob['Commodity Name']==crop_name.upper())]
        data_sob.to_csv(f'{output_dir}/{crop_name}_sob_all.csv', index=False)
    else:
        data_sob = pd.read_csv(f'{output_dir}/{crop_name}_sob_all.csv',dtype={'FIPS':str},low_memory=False)
    return data_sob


def load_rma_loss_ratio(crop_name='corn', level='county'):
    data_sob = load_rma_sob_all(crop_name=crop_name,rerun=False)    

    if level == 'county':
        loss_ratio = data_sob.groupby(['FIPS', 'Commodity Year']).sum()['Indemnity Amount']\
        /data_sob.groupby(['FIPS', 'Commodity Year']).sum()['Total Premium Amount']

    if level == 'national':
        loss_ratio = data_sob.groupby(['Commodity Year']).sum()['Indemnity Amount']\
        /data_sob.groupby(['Commodity Year']).sum()['Total Premium Amount']
    
    return loss_ratio.reset_index().rename(columns={0:'Loss_ratio'})

def load_rma_loss_ratio_cause(crop_name='corn',rerun=True):
    if rerun:
        # load RMA loss and SOB data
        data_loss = load_rma_loss_all(crop_name=crop_name,rerun=False)
        data_sob = load_rma_sob_all(crop_name=crop_name,rerun=False)

        # Indemnity from RMA loss data (sum by FIPS, Year)
        data_loss_cause = data_loss.groupby(['FIPS','Commodity Year','Damage Cause Description']).sum()['Indemnity Amount']
        data_loss_cause_sum = data_loss.groupby(['FIPS','Commodity Year']).sum()['Indemnity Amount']

        
        # Indemnity from RMA SOB data (sum by FIPS, Year)
        data_sob_sum = data_sob.groupby(['FIPS','Commodity Year']).sum()
        # data_loss_sum = data_loss.groupby(['FIPS','Commodity Year']).sum()
        # print(data_sob_sum)

        data_loss_cause_percent = data_loss_cause.unstack('Damage Cause Description'). \
            div(data_loss_cause_sum, axis=0).stack('Damage Cause Description'). \
            reorder_levels(data_loss_cause.index.names)        

        # Loss ratio from RMA SOB data 
        loss_ratio = load_rma_loss_ratio(crop_name=crop_name, level='county')
        loss_ratio.set_index(['FIPS', 'Commodity Year'], inplace=True)
        
        # Loss ratio disaggregated into different causes
        loss_ratio_cause = data_loss_cause_percent.unstack('Damage Cause Description'). \
            mul(loss_ratio['Loss_ratio'], axis=0).stack('Damage Cause Description')

        # Merge all these variables 
        loss_ratio_cause = loss_ratio_cause.reset_index(). \
            rename(columns={0:'Loss ratio by cause'}). \
            merge(loss_ratio.reset_index().rename(columns={'Loss_ratio':'Loss ratio all cause'})).\
            merge(data_loss_cause_percent.to_frame().reset_index(). \
                rename(columns={0:'Cause percent'})). \
            merge(data_loss_cause.reset_index(). \
              rename(columns={'Indemnity Amount':'Indemnity Amount by cause'})). \
            merge(data_sob_sum['Indemnity Amount'].to_frame().reset_index(). \
              rename(columns={'Indemnity Amount':'Indemnity Amount sum SOB'})). \
            merge(data_loss_cause_sum.to_frame().reset_index(). \
              rename(columns={'Indemnity Amount':'Indemnity Amount sum loss'}))
        
        loss_ratio_cause['Damage Cause Description'] = loss_ratio_cause['Damage Cause Description'].\
            replace('Excess Moisture/Precipitation/Rain','Excess Moisture/Precip/Rain')
        
        loss_ratio_cause.to_csv(f'{output_dir}/{crop_name}_loss_ratio_cause.csv', index=False)
    else:

        loss_ratio_cause = pd.read_csv(f'{output_dir}/{crop_name}_loss_ratio_cause.csv', dtype={'FIPS':str})
    return loss_ratio_cause

def load_rma_loss_ratio_cause_month(crop_name='corn',rerun=False):
    if rerun:
        # load RMA loss and sob data
        data_loss = load_rma_loss_all(crop_name=crop_name,rerun=False)
        data_loss['Damage Cause Description'] = data_loss['Damage Cause Description'].\
            replace('Excess Moisture/Precipitation/Rain','Excess Moisture/Precip/Rain')
      
        # Indemnity from RMA loss data (sum by FIPS, Year)
        data_loss_cause = data_loss.groupby(['FIPS','Commodity Year','Damage Cause Description',
                                             'Month of Loss Abbreviation']).sum()['Indemnity Amount']
        data_loss_cause_sum = data_loss.groupby(['FIPS','Commodity Year','Damage Cause Description'])\
            .sum()['Indemnity Amount']

        # Calculate percentage of indemnity loss by cause
        data_loss_cause_percent = data_loss_cause.unstack('Month of Loss Abbreviation'). \
            div(data_loss_cause_sum, axis=0).stack('Month of Loss Abbreviation'). \
            reorder_levels(data_loss_cause.index.names)
        
        # Based on loss_ratio_cause to get one level deeper to month
        loss_ratio_cause = load_rma_loss_ratio_cause(crop_name,rerun=False)
        loss_ratio_cause = loss_ratio_cause.set_index(['FIPS', 'Commodity Year', 'Damage Cause Description'])

        # Loss ratio disaggregated into different causes and different months
        loss_ratio_cause_month = data_loss_cause_percent.unstack('Month of Loss Abbreviation'). \
            mul(loss_ratio_cause['Loss ratio by cause'], axis=0).stack('Month of Loss Abbreviation')
        loss_ratio_cause_month = loss_ratio_cause_month.to_frame('Loss ratio by cause and month').\
            reset_index().rename(columns={'Commodity Year':'Year'})  
        loss_ratio_cause_month['Damage Cause Description'] = loss_ratio_cause_month['Damage Cause Description'].\
            replace('Excess Moisture/Precipitation/Rain','Excess Moisture/Precip/Rain')
        
        loss_ratio_cause_month.to_csv(f'{output_dir}/{crop_name}_loss_ratio_cause_month.csv')
        print('Rerun function, file RMA_loss_ratio_cause_month.csv saved')

    else:    
        loss_ratio_cause_month = pd.read_csv(f'{output_dir}/{crop_name}_loss_ratio_cause_month.csv',dtype={'FIPS':str})
        print('Do not rerun function, load data from file RMA_loss_ratio_cause_month.csv')
    
    return loss_ratio_cause_month

if __name__ == '__main__':

    data_dir   = '../../data/USDA-RMA/'
    start_year = 1989
    end_year   = 2016
    
    output_dir = '../output/usda_rma_data'
    os.makedirs(output_dir, exist_ok=True)

    name_crop = ['corn','soybeans','wheat']
    for ic in range(len(name_crop)): 
        crop_name = name_crop[ic]
        print(f'crop name: {crop_name}')
        load_rma_loss_all(crop_name,rerun=True)
        load_rma_sob_all(crop_name,rerun=True)
        load_rma_loss_ratio_cause(crop_name,rerun=True)
        load_rma_loss_ratio_cause_month(crop_name,rerun=True)
        print(f'RMA data for: {crop_name} saved! \n')


