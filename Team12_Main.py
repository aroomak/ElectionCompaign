# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:34:08 2021

@author: Alan
"""

import pandas as pd
import numpy as np
from datetime import datetime
import math


ourteam='Team 12'
out_col_name=['Out_0','Out_1','Out_2']

import Team12_config

#################################################
## Initialization
#################################################

## Getting File Paths
CPath_Data=Team12_config.Merged_df_path
CPath_GpList=Team12_config.Group_List_Used
CPath_Out=Team12_config.Assignment_Folder_Path

print('Merged_Data_Path :',CPath_Data)
print('Clustering_Result_Path :',CPath_Data)
print('Action_Out_Path :',CPath_Data)

## Importing all records and intialize basic parameters
Data=pd.read_csv(CPath_Data)
print(Data.columns)
version='v4'   
  
List_Munici=pd.read_csv (CPath_GpList)
Count_Munici=List_Munici['Municipality'].count()
List_Munici_Names=List_Munici['Municipality'].unique()


## Statistics of the records we have
## Number of Records Per Group
RecordsPerGroup=Data.groupby('Groups')['Groups'].count()
print('Number of Records Per Group:',RecordsPerGroup)
TotalGroupNo=Data['Groups'].max()+1
Newest_Date=Data['Date'].max()
newest_day=Data['Day'].max()
list_date=Data['Date'].unique()
    


#################################################
def Generate_Action_Submission_Frame():
    List_Munici=pd.read_csv (CPath_GpList) 
    df_Frame=List_Munici
    
    ## Another Version: To Include the action matrix
    # action_table=pd.DataFrame(np.zeros((Count_Munici,3)))
    # df_Frame=pd.concat([df_Frame,action_table], axis=1)
    
    return(df_Frame)

#################################################    
 ## For group "group_no", find the 'best action' row vector
def find_best_action(group_no):
    this_condition=(Data['Day'] > newest_day -4) & (Data['Groups']==group_no) ## Condition : records within 5 newest day and is in this group
    Group=Data[this_condition]
    
    Q_Range=0.95 ## only top 5 % vote % record are used
    Q_Value=Group[ourteam].quantile(Q_Range)
    
    ## Data Selection
    top_5p=pd.DataFrame(Group[Group[ourteam]>=Q_Value]) ## the record of the top 5% 
    top_5p.loc[:,'rank']=np.array(top_5p[ourteam].rank())
    
    ## if the mean vote% is too low -> it means historical records is not meaningful -> reset to [1,1,1]
    ## otherwise,find the best action from last 4 days records
    mean_vote=top_5p[ourteam].mean()
    vote_thereshold=0.00 ## mean voting % thereshold
    
    ## deciding if the top 5% finds a direction or not 
    ## V1 by looking at variance of each action
    # top_5_var=np.array(top_5p.loc[:,'0':'2'].var())

    
    ## V2 by counting unique municipalities and the vote % >2%
    good_vote_per=0.02
    count_good=(top_5p[ourteam]>=good_vote_per).sum()
    count_total=len(top_5p[ourteam].index)
    count_unique_municipalities=len(top_5p['Municipality'].unique())
    ratio_good_vote=count_good/count_total
    ratio_enought_municipality=count_unique_municipalities/count_total
    ratio_combine=ratio_good_vote*ratio_enought_municipality
    
    ## Decide if this cluster need small exploration (1%) or big exploration (5%)
    if  ratio_combine<0.5:
        bo_explore=True
    else:
        bo_explore=False
    explore_decision=np.append(ratio_combine,bo_explore)

    if mean_vote>= vote_thereshold:
        ## use the 'weighted_average(top_5p)' to get the 'fine-tune vector' and normalize it before saving it
        mean_action=Normalize_Action(weighted_average(top_5p,10)) 
        
    else:
        mean_action=Normalize_Action([1,1,1])
    out=np.append(mean_action,mean_vote)
    out=np.append(out,explore_decision)
    
    ## v1 : take action vector of the max vote %
    # row_max=Group[ourteam].idxmax()
    # action_max=np.array(Group.loc[row_max,'0':'Key']).T
    return out


#################################################
## Calculate the average vote % for each group and return a list
def average_vote_percentage():
    ## Two Version: _N is only consider records from the newest date, without _N consider all records
    ##2021-0112 : We change from taking average to median ,to minimize the effect of outliers
    l_gpmean=[]
    l_gpmean_N=[]
    for group_no in range(0,TotalGroupNo):
        Group=Data[Data['Groups']==group_no]
        Group_Newest=Data[(Data['Groups']==group_no) & (Data['Date']==Newest_Date)]
        Gpmean=Group[ourteam].median()
        Gpmean_N=Group_Newest[ourteam].median()
        l_gpmean.append(Gpmean)
        l_gpmean_N.append(Gpmean_N)
    return {'avg_vote':l_gpmean,'avg_vote_N':l_gpmean_N}

#################################################
## Decide on the level of exploration 1% or 5%
def decide_on_var(best_action):
    low_var=0.01 
    high_var=0.05
    
    
    ## Version 1
    ## If the mean vote % is lower than thereshold, we should use higher variance to generate more diverse set of action_
    ## If on the other case, we want to have smaller variation, to fine tune the action parameters
    thereshold=0.05
    
    ## Two version: use mean from all records or just the newest day
    gpmean=pd.DataFrame(average_vote_percentage(),columns=(['avg_vote','avg_vote_N']))
    
    Use_all_record=False
    
    if Use_all_record:
        col_used='avg_vote'
    else:
        col_used='avg_vote_N'
    # gpmean.loc[gpmean[col_used]<thereshold,'var']=high_var
    # gpmean.loc[gpmean[col_used]>=thereshold,'var']=low_var
    
    
    ## Version 2
    ## Already decided in function "find_best_action"
    gpmean.loc[:,'exploration_needed']=best_action['exploration_needed']
    gpmean.loc[best_action['exploration_needed']==1,'var']=high_var
    gpmean.loc[best_action['exploration_needed']==0,'var']=low_var
    return gpmean

#################################################
def Normalize_Action(action):
    total=sum(action)
    return np.array([x/total for x in action])



################################
## Function for fine-tuning the best action
################################
def df_to_datetime(column_date):
    return pd.to_datetime(column_date, format='%Y-%m-%d')

def Day_Difference(df_date,target_date):
    target_date=datetime.strptime(target_date,'%Y-%m-%d')
    df_date=df_to_datetime(df_date)
    print (df_date-target_date)
    
 #######################################################################
## find the weighted average of the action vector
def weighted_average(top,Rank_Weight): ## Weight on rank relative to the newness
    column=['0','1','2']
    action=[]
    # Rank_Weight=0.5 ## we want to give more weight to recent records
    weighted_total=top['Day'].sum()+top['rank'].sum()*Rank_Weight
    for col in column:
        Day_Dot=top.loc[:,col].dot(top['Day'])
        Rank_Dot=top.loc[:,col].dot(top['rank'])
        weighted_sum=Day_Dot+Rank_Dot*Rank_Weight
        action.append(weighted_sum/weighted_total)
     
    return action

#######################################################################
## Find an action vector for fine-tuning the cluster mean
## this action vector represents the direction that the group should be adjusted
def Municipality_Good_Action(m_name):
    if type(m_name)== str:
        df_M=Data[Data['Municipality']==m_name]
    else:
        df_M=Data[Data['Groups_Specific']==m_name]
    Q_Range=0.8 ## only top 20 % vote % record are used
    Q_Value=df_M[ourteam].quantile(Q_Range)
    
    ## Data Selection
    top_10p=pd.DataFrame(df_M[df_M[ourteam]>=Q_Value]) ## the record of the top 20% 
    top_10p.loc[:,'rank']=np.array(top_10p[ourteam].rank())
    
    ## use the 'weighted_average(top_10p)' to get the 'fine-tune vector' and normalize it before saving it
    mean_action=Normalize_Action(weighted_average(top_10p,0.5)) 

    return df_M,top_10p,mean_action

#######################################################################
## Find the fine tune vector for the sub-group
## Can go by municipality or the specific groups
## Idea: for each municipality/group, go to function 'Municipality_Good_Action' and it will return a best action vector for this group
def best_action_finetune(Version): ## Option:'Municipality' or 'SGroup'
    dict_out={}
    if Version=='Municipality':
        rename_index={'index':'Municipality',0:'FT_0',1:'FT_1',2:'FT_2'}  ## Setting the index for output dataframe
        for municipality in List_Munici_Names:      #Loop through all municipalities
            df,top_10,dict_out[municipality]=Municipality_Good_Action(municipality)  ## Get the vector from the Good_Action function
    elif Version=='SGroup':
        print(Version)
        List_Specific_Groups=sorted(Data['Groups_Specific'].unique())
        rename_index={'index':'Groups_Specific',0:'FT_0',1:'FT_1',2:'FT_2'}
        for this_group in List_Specific_Groups: #Loop through all specific groups
            df,top_10,dict_out[this_group]=Municipality_Good_Action(this_group) ## Get the vector from the Good_Action function
        
    df_out=pd.DataFrame(dict_out).T.reset_index() ## put the output into a datafram
    df_out=df_out.rename(columns=rename_index)  ## reset the column names
    return  df_out


################################
## Main CBA Running Function
################################
def Cluster_Best_Action(Bo_Var):
    
    #########################################
    # Find Current 'Best Action' for all groups
    best_action=pd.DataFrame(np.zeros((TotalGroupNo,6)))
    for this_group in range(0,TotalGroupNo):    
        best_action.iloc[this_group,:]=find_best_action(this_group)
    best_action=best_action.rename({3:'top_5p_mean',4:'combine_direction_ratio',5:'exploration_needed'},axis=1)
    best_action['Groups']=np.array(range(0,TotalGroupNo))
    df_Output=Generate_Action_Submission_Frame().merge(best_action,left_on='Groups',right_on='Groups',how='left')
    
    #########################################
    ## Generate the fine_tune vectors
    df_fine_tune=List_Munici.merge(best_action_finetune('SGroup')) ## Option:'Municipality' or 'SGroup'
    df_Output=df_Output.merge(df_fine_tune,how='left')
    
    gba_col_name=[0,1,2] ## group best action column name
    ft_col_name=['FT_0','FT_1','FT_2'] ## fine-tune action column name
    adj_col_name=['adj_0','adj_1','adj_2'] ## fine-tune action column name
    
    #########################################
    ## Apply fine tuning to corresponding subgroup before adding in the exploration component, and put the result in "adj col"
    finetune_before_random=True
    if finetune_before_random:
        df_Output[adj_col_name]=(df_Output[gba_col_name].values+df_Output[ft_col_name].values)/2
    else:
        df_Output[adj_col_name]=df_Output[gba_col_name]
    
    mod_col_name=[]
    

    #########################################    
    ##use function 'decide_on_var()' to decide on the variance selected
    l_var=decide_on_var(best_action)
    
    #########################################
    ###For each municipality, Take the adj col to add randomness
    for i in adj_col_name: ## for each action column
        column_name='Out_'+i[4]
        mod_col_name.append(column_name)
        
        ## for each municipality in the action column i
        this_rand=np.zeros(Count_Munici)
        for rows in df_Output.index:
            
            ## find the corresponding group,group_mean,group_selected_variance
            rows_group=df_Output.loc[rows,'Groups']
            this_scale=l_var.loc[rows_group,'var']
            this_mean=df_Output.loc[rows,i]
            
            ## generate the random number of a normal distribution with group mean,designated variance from l_var
            this_rand[rows]=np.random.normal(this_mean,this_scale, size=1)
            
        ## if the generated number is negative, we take the absolute value of it
        this_rand[this_rand<0]=abs(this_rand[this_rand<0])
        
        ## 2 mode is avilable , Bo_Var == FALSE -> no randomness is added,Bo_Var == TRUE -> randomness is added
        if Bo_Var:
            df_Output[column_name]=this_rand
        else:
            df_Output[column_name]=df_Output[i]
            

    #########################################
    ## Normalize the row -> sum of all rows =1
    df_Output['Out_Total']=df_Output.loc[:,mod_col_name].sum(axis=1)
    for column_name in mod_col_name:
        df_Output[column_name]=df_Output[column_name]/df_Output['Out_Total']
        
    #########################################
    ## update the 'Mod Total' before updating
    df_Output['Out_Total']=df_Output.loc[:,mod_col_name].sum(axis=1)
    return df_Output,l_var,best_action

################################
## Function for exporting actions
################################

def assignment_check_and_export(export_df):
    ## Checking the Dimension of export dataframe
    print('\nStart Export Checking')
    print(export_df.shape)
    
    ##Checking if the sum of row is equal to 1
    pd.set_option("max_rows", None)
    checking_100=abs(export_df.sum(axis=1) -1)>0.001 
    Violate_100=checking_100.sum()
    print('Number of incorrect rows:',Violate_100)
    pd.set_option("max_rows", 10)
    
    ##Checking if all values are larger than 0
    df_CL=export_df>=0 
    column_larger=df_CL.sum()==Count_Munici
    print('All Values > 0:',all(column_larger==True))
    
    ## only if both condition are met,export an txt file
    if Violate_100 ==0 and all(column_larger==True) :
        date=datetime.now().strftime('%Y-%m%d-%H%M')
        filename=CPath_Out+'Sample_team12.txt'
        
        ##Export of Assignment
        export_df.to_csv(filename,header=False,index=False)
        print('file exported successfully to:',filename)




##############################################################################
##Main
##############################################################################
def Main_run(): 
    
    ## To run the CBA function 
    df_Cluster_Best_Action,today_var,best_action_summary=Cluster_Best_Action(True)
    ## Perform Export
    assignment_check_and_export(df_Cluster_Best_Action.loc[:,out_col_name])

