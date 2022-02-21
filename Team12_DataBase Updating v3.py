# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:31:20 2021

@author: 
"""

## Goal of Code: 
    ## 1.Automatically update the 'all_his_campaign.csv' and 'all_his_polls.csv'
    ## ie. extract all past records and append into 1 big file
    ## 2.Connecting the action taken and the corresponding result into 1 row for further analysis and prediction

##path of the folder for the two source
campaign_folder_path=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - CaseStudy_period3\Data\Submitted Campaign'+'\\'
historical_polls_folder_path=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - CaseStudy_period3\Data\Historical Polls'+'\\'



import pandas as pd
import numpy as np
import glob
import os
from os import listdir
import datetime

## To get the municipality list
Group_List_ExportPath1=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - CaseStudy_period3\Data\2020-0105_Groupingv1.csv'
Group_List_ExportPath2=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - CaseStudy_period3\Data\2020-0108_Groupingv2.csv'
Group_List_ExportPath3=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - CaseStudy_period3\Data\2020-0114_Groupingv3.csv'
Group_List_ExportPath4=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - CaseStudy_period3\Data\2020-0119_Groupingv4.csv'

## Different Clustering Result Option
## v1 C19_without normalization
## v2 C19_with normalization
## v3 C6_tsne+GM/OPTICS
## v4 C6_tsne+H-clustering
this_clustering_version='v4'
Group_List_Path={'v1':Group_List_ExportPath1,'v2':Group_List_ExportPath2,'v3':Group_List_ExportPath3,'v4':Group_List_ExportPath4}
List_Munici=pd.read_csv (Group_List_Path[this_clustering_version])


def df_to_datetime(column_date):
    return pd.to_datetime(column_date, format='%Y-%m-%d')

## Calculate the day difference
def Day_Difference(df_date,target_date):
    # target_date=datetime.strptime(target_date,'%Y-%m-%d')
    # df_date=df_to_datetime(df_date)
    raw_days=(df_date-target_date).dt.days
    week=np.floor(raw_days/7)
    return raw_days-week*2+1

## function to get all txt in the folder
def find_txt_filenames( path_to_dir, suffix=".txt" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

## function to open all CAMPAIGN txt and append into one big dataframe
def get_all_his_campaigns(folder_path):
    initiated=False
    for filename in find_txt_filenames(folder_path):
        filepath=folder_path+filename
        date=datetime.datetime.strptime(filename[:9],'%Y-%m%d')
        
        if initiated:
            thisdf=pd.read_csv(filepath,header=None)
            thisdf.loc[:,'Date_C']=date
            thisdf.loc[:,'Municipality']=List_Munici['Municipality']
            df=pd.concat([df,thisdf])
        else:
            df=pd.read_csv(filepath,header=None)
            df.loc[:,'Date_C']=date
            df.loc[:,'Municipality']=List_Munici['Municipality']
            initiated=True
    return(df)

## function to open all POLL RESULT txt and append into one big dataframe
def get_all_his_polls(folder_path):
    initiated=False
    for filename in find_txt_filenames(folder_path):
         if "2021" in filename:
            filepath=folder_path+filename
            date=datetime.datetime.strptime(filename[:9],'%Y-%m%d')
            if initiated:
                thisdf=pd.read_csv(filepath)
                thisdf.loc[:,'Date_P']=date
                df=pd.concat([df,thisdf])
            else:
                df=pd.read_csv(filepath)
                df.loc[:,'Date_P']=date
                initiated=True

    return(df)

def merged_records(All_Campaign,All_Polls):
    # All_Campaign=pd.read_csv(all_campaign_path)
    # All_Polls=pd.read_csv(all_hispolls_path)
    All_Campaign['Key']=All_Campaign.Municipality+All_Campaign.Date_C.apply(lambda x: x.strftime('%Y-%m-%d'))
    All_Polls['Key']=All_Polls.Municipality+All_Polls.Date_P.apply(lambda x: x.strftime('%Y-%m-%d'))

    
    ## Merging the Action and Polls Result
    Merged_df=All_Campaign.merge(All_Polls,left_on=['Key'],right_on=['Key'],how='inner')
    Merged_df=Merged_df.drop(columns=['Date_C','Municipality_x']).rename(columns={'Date_P':'Date','Municipality_y':'Municipality'})
    
    ## Add the clustering info
    Merged_df=Merged_df.merge(List_Munici,how='left')
    
    ## Calculate how new the data is
    Start_Date=Merged_df['Date'].min()
    Merged_df['Day']=Day_Difference(Merged_df['Date'],Start_Date)
    return Merged_df

## Running the two functions
all_hiscampaign=get_all_his_campaigns(campaign_folder_path)
all_hispolls=get_all_his_polls(historical_polls_folder_path)
Merged_records=merged_records(all_hiscampaign,all_hispolls)

group_no=len(Merged_records['Groups'].unique())
Cluster_Size='C'+this_clustering_version

all_campaign_path=campaign_folder_path+'all_his_campaign_'+Cluster_Size+'.csv'
all_hispolls_path=historical_polls_folder_path+'all_his_polls_'+Cluster_Size+'.csv'
Merged_df_path=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - CaseStudy_period3\Data\Merged_Record.csv'
Merged_df_specific_path=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - CaseStudy_period3\Data\Merged_Record_'+Cluster_Size+'.csv'

## Export the pandas dataframe to replace the old main csv file
all_hiscampaign.to_csv(all_campaign_path,index=None)
all_hispolls.to_csv(all_hispolls_path,index=None)
Merged_records.to_csv(Merged_df_path,index=None)
Merged_records.to_csv(Merged_df_specific_path,index=None)
