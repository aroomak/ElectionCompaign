#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jan 13 10:35:22 2021

@author: aram
"""

### Importing the merged dataset 

import pandas as pd
import numpy as np
from random import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Path where the Merged Record exists 
path = "//home/aram/Desktop/Draft/CaseStudy/"
# Path to save plots and final DataFrame
mergd = pd.read_csv(path+'Merged_Record.csv')
team_cols = [col for col in mergd if col.startswith('Team')]
action_cols = ['0','1','2'] 
needed_cols = action_cols + team_cols    # because 'team_cols' is a list not an array
#needed_cols = ['0','1','2'] + team_cols


############################################################
###      Values to be define from the begining:          ###
threshold = 0.05         # selecting best results          #
sample_size = 100000     #Sample size                      #
sample_size_wv = 100000  #Sample size for Weighted Avg Vec #
n_best_act = 20       # Number of best actions to plot     #
n_trials = 1          # Number of trials for every cluster #  
############################################################


#################################  Functions  ##############################

## To select a cluster and separate Train/Test datasets
def data_prp(i):
    ## Select Cluster i
    merged_g = mergd[mergd['Groups'] == i]
    # Select only actions with more than 5% votes 
    merged_g = merged_g.loc[merged_g['Team 12'] > threshold ]
    
    
    X = merged_g[action_cols]
    Y = merged_g[team_cols]
    #Y = np.array(merged_g_01['Team 12'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
    
    # Normalization
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std    
    return (X_train, Y_train, X_test, Y_test)


### Non Linear Regression with DNN
## Define the model
def nn_model():
    model = Sequential()
    # since we have 3 actions, input dimension = 3
    model.add(Dense(64, input_dim=3, activation='relu', name='hl1'))
    #model.add(Dense(32, activation='relu', name='hl2'))
    #model.add(Dense(32, activation='relu', name='hl3'))
    # Because its a regression case, wa want the real value and use 'linear' activation function
    model.add(Dense(16, activation='linear', name='opl'))
    model.compile(optimizer='adam', loss='LogCosh', metrics=['mse','mae'])
    model.summary()
    return model

## Fitting/Training the Model
# Also We mention which ratio of the Dataset is for Testing/Validation
def fitting_model(x,y,i):
    history = nn_model().fit(x, y, epochs=150, batch_size=50, verbose=1, validation_split=0.1)
    return ((history.history['loss']), (history.history['val_loss']))

## Plot Training performance 
def plot_training_performance(los, losVal, i, r):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # Left plot
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(los, 'r')
    ax.plot(losVal, 'g')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    plt.grid(True)
    plt.style.use('seaborn')
    plt.legend(['epochs','loss'], loc='best')
    # Right plot
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(los, 'r')
    ax.plot(losVal, 'g')
    ax.set_xlabel('epochs')
    #ax.set_ylabel('loss')
    ax.set_ylim(0.0, 0.04)
    fig.suptitle('Model loss - training_Performance Group %d Round %r' %(i, r), y=0.95)
    fig.savefig(path+'02 Training_Performance Group%d round %d.png' %(i, r), dpi=300)
    plt.grid(True)
    plt.style.use('seaborn')
    plt.legend(['epochs','loss'], loc='best')
    plt.show()


## Evaluate/Testing
def testing_model(x,y):
    loss_nn, mse_nn, mae_nn = nn_model().evaluate(X_test, Y_test)
    print('Loss on test data: ', loss_nn)
    print('Mean squared error on test data: ', mse_nn)
    print('Mean absolute error on test data: ', mae_nn)


## Prediction
def predicting_model(x):
    Xnew = np.array([x])
    ynew= nn_model().predict(Xnew)
    #print("x_new=%s, \n Predicted=%s" % (Xnew[0], ynew[0]))
    return ynew[0]

## To generate a DataFrame with random values 
def gen_rand_weights():
    num=[]
    for i in range(0,3):
        num.append(random())
    total=sum(num)
    return [x/total for x in num]

def random_sample_gen(sample_size):
    random_sample = pd.DataFrame(np.zeros((sample_size,3)))
    for i in range(0,sample_size):
        random_sample.iloc[i,:] = gen_rand_weights()
    return random_sample


## To get the size of every cluster/group (number of municipalities)
def get_group_size(g):
    dates = sorted(mergd.Date.unique())
    mergd_size = mergd[mergd['Date'] == dates[0]]
    mergd_size = mergd_size[mergd_size['Groups'] == g]
    return len(mergd_size)
    

## Get the top N yHat for every Cluster 
def top_n_yHat(random_sample, prediction_var, cluster_size):
    RandomX_PredictedY = pd.DataFrame(random_sample)
    RandomX_PredictedY['y'] = pd.DataFrame(prediction_var)
    return RandomX_PredictedY.nlargest(cluster_size,'y')


## Creating the final dataset .v3
def create_final_result(final_predicted_dataset, top_n_yHat_values, i):
    grp = np.ones((len(top_n_yHat_values),1))
    grp.fill(i)
    top_n_yHat_values = (top_n_yHat_values).to_numpy()
    #merge top_n_yHat and groups 
    base11 = np.concatenate([top_n_yHat_values, grp], axis=1)
    final_predicted_dataset = np.append(final_predicted_dataset, base11, axis=0)
    return final_predicted_dataset

## Creating the final dataFrame with headers
def create_dataframe_final_result(final_predicted_dataset):
    cols = ['a1','a2','a3', 'yHat', 'Group']
    final_predicted_DataFrame = pd.DataFrame(final_predicted_dataset, columns=cols)
    final_predicted_DataFrame = final_predicted_DataFrame.iloc[1:]
    return final_predicted_DataFrame

## Plotting the top N yHat vs three actions 
def ploting3d(predicted_data, i, r): 
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # Left plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    img = ax.scatter(predicted_data[0], predicted_data[1], predicted_data[2], c=predicted_data['y'], cmap='coolwarm')
    
    ax.set_xlabel('a1')
    ax.set_ylabel('a2')
    ax.set_zlabel('a3')
    ax.view_init(30, 35)
    ax.set_zlim(0, 1)
    # Right plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    img = ax.scatter(predicted_data[0], predicted_data[1], predicted_data[2], c=predicted_data['y'], cmap='coolwarm')
    ax.set_xlabel('a1')
    ax.set_ylabel('a2')
    ax.set_zlabel('a3')
    ax.view_init(60, 35)
    fig.colorbar(img)
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.set_zlim(0.0, 1.01)
    fig.suptitle('Top 500 yHat in Group %d round %d'%(i, r), y=0.95)
    fig.savefig(path+'03 Prediction_plot Group%d round%d.png' %(i, r), dpi=300)
    plt.show()

## plot top actions 
def ploting3d_2(predicted_data, i): 
    fig = plt.figure(figsize=plt.figaspect(0.5))
    # Left plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    img = ax.scatter(predicted_data['a1'], predicted_data['a2'], predicted_data['a3'], c=predicted_data['y'], cmap='coolwarm')  
    ax.set_xlabel('a1')
    ax.set_ylabel('a2')
    ax.set_zlabel('a3')
    ax.view_init(30, 35)
    ax.set_zlim(0, 1)
    # Right plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    img = ax.scatter(predicted_data['a1'], predicted_data['a2'], predicted_data['a3'], c=predicted_data['y'], cmap='coolwarm') 
    ax.set_xlabel('a1')
    ax.set_ylabel('a2')
    ax.set_zlabel('a3')
    ax.view_init(60, 35)
    fig.colorbar(img)
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.set_zlim(0.0, 1.01)
    fig.suptitle('Top 10 Actions and Votes in Group %d'%i, y=0.95)
    fig.savefig(path+'01 ActionsGroup_%d.png'%i, dpi=300)
    plt.show()


## Calculate the wight of every vote based on the accurance day and vote result
def Weighted_Avg_Vec(topVote2):
    actions = np.array(topVote2[['a1','a2','a3']] )     #Getting actions
    days = np.array(topVote2[['Day']])                  #Getting corresponing days
    weightedAvgVec = np.multiply(actions, days)         #A.Day = (a1, a2, a3) * days
    weightedAvgVec = weightedAvgVec.sum(axis=0)         #sum(A.Day)
    days_sum = days.sum(axis=0)                         #sum(days)
    # sum(A.Day) / sum(days) => Weighted Average Vector 
    weightedAvgVec = np.true_divide(weightedAvgVec, days_sum)
    weightedAvgVec = pd.DataFrame(weightedAvgVec).transpose()
    return weightedAvgVec

## Generate the random sample size based on the Weighted Average Vector
def random_sample_WV_based(weightedAvgVec):
    RandomSample_w = np.zeros((sample_size_wv,1))
    for i in range(3):
        #TODO: define dynamic range for variance than defining manually 
        a = np.random.normal(weightedAvgVec[i], 0.02, (sample_size_wv,1))
        RandomSample_w = np.concatenate([RandomSample_w, a], axis=1)
        RandomSample_w = pd.DataFrame(RandomSample_w)
        #RandomSample_wav.loc[:,1:3]
    return RandomSample_w.loc[:,1:3]

###################################################################
##          Min/Max number of samples for designing NN           ##
###################################################################
maxlen = 0
minlen = 1000
n = sorted(mergd.Groups.unique())
#print("Number of Clusters/Groups: ",n)
for g in n:
    merged_g = mergd[mergd['Groups'] == g]
    if len(merged_g) < minlen:
        minlen = len(merged_g)
    if len(merged_g) > maxlen:
        maxlen = len(merged_g)
    minmax = [[minlen, maxlen]]
print(" Maximum and Minimum size of samples are: ", (minmax))
###################################################################
##             Running the Program  -   Automation               ##
###################################################################
### Run for the merged Dataset
# although we know no of clusters, if we want to get range of clusters: 

#############@@@@@@ for loop to run the whole code automatically 

# to get the list of clusters 
n = sorted(mergd.Groups.unique())
#n = [1, 4, 13]
print("Number of Clusters/Groups: ",n)
final_predicted_dataset = np.zeros((1,5)) 

for g in n:
    print(" @Group %d " %g)
    #
    merged_g = mergd[mergd['Groups'] == g]
    # Select only actions with more than 5% votes 
    merged_g = merged_g.loc[merged_g['Team 12'] > 0.05 ]
    # Select top actions/votes in every cluster
    topVote = merged_g.nlargest(n_best_act, 'Team 12')
    needed_cols2 = ['Team 12', 'Day', 'Key', 'Groups']
    needed_cols2 = action_cols+needed_cols2
    topVote = topVote[needed_cols2]
    #topVote2 = topVote[(action_cols+['Team 12'])]
    topVote2 = topVote.rename(columns={"0": "a1", "1": "a2", "2": "a3", "Team 12": "y"})
    # Plot history of best actions 
    ploting3d_2(topVote2, g)
    
    #get the Weighted Average vector
    w_avg_vec = Weighted_Avg_Vec(topVote2)
    #generate the random Sample based on the Weighted Average vector
    RandomSample_wav = random_sample_WV_based(w_avg_vec)

    for r in range(n_trials):
        print(" @ Round %d" %r)
        # i => Cluster/Group Number
        ## Fitting and Testing the model
        X_train, Y_train, X_test, Y_test = data_prp(g)
        los, losval = fitting_model(X_train, Y_train, g)
        #Plotting the Training Performance of the NN 
        plot_training_performance(los, losval, g, r)
        ## Testing the trained model
        testing_model(X_test, Y_test)
            
        ## Generating Random Dataset for prediction
        print(" ================ [ Predicting Group: %d round %d ] ===.... .  .  .  .   .    ." %(g ,r))
           
        random_sample = random_sample_gen(sample_size)
    
        ## Prediction
        prediction_var = predicting_model(random_sample)
        #prediction_var = predicting_model(RandomSample_wav)
                
        prediction_var_12 = prediction_var[:,11]
        
        ## Get top N values of predicted Y Hat 
        #top_n_yHat_values =top_n_yHat(random_sample, prediction_var, get_group_size(i))
        RandomX_PredictedY = pd.DataFrame(random_sample)
        RandomX_PredictedY['y'] = pd.DataFrame(prediction_var_12)
        top_n_yHat_values = RandomX_PredictedY.nlargest(get_group_size(g),'y')
        
        
        ## Create accumulated dataset for all groups
        print(" ================ [ Generating Dataset ] ===.... .  .  .  .   .    .")
        final_predicted_dataset = create_final_result(final_predicted_dataset, top_n_yHat_values, g)
        
        
        ## Plotting the yHat Vs actions #Top500 yHats
        print(" ================ [ Plotting Group: %d Round %d ] ===.... .  .  .  .   .    ." %(g ,r))
        #ploting3d(top_n_yHat_values)
        #RandomX_PredictedY.nlargest(get_group_size(g),'y')
        #ploting3d(RandomX_PredictedY.nlargest(500,'y'),g)
        ploting3d(RandomX_PredictedY, g, r)
        
    
    print("The Regression Prediction finished")
    final_DataFrame_toSave = create_dataframe_final_result(final_predicted_dataset)
    final_DataFrame_toSave.to_csv (path+'DataFrame_RegressionNN_Result.csv', index = False, header=True)
    print("The Final DataFrame has been created")
    
    
