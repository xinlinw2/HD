import HDFunctions
import numpy as np
import pickle
import pandas as pd
import os
import time
from sklearn.preprocessing import MinMaxScaler
import warnings
import tracemalloc
warnings.filterwarnings("ignore")
#tracemalloc.start()

#dimensionality of the HDC model
D = 10000 #10K
#number of level hypervectors
nLevels =4
#number of retraining iterations
n =1
#"""
Per_=0.8
Path_sorce_="C:\\Users\\Xinlin\\Desktop\\HD_test\\data_sources\\32_days_normalized.csv"
#Path_sorce_="uci_normalized.csv"
#Path_sorce_="ecg_normalized_3.csv"
#Path_sorce_="ECG_database.csv"
#Path_sorce_="ECG_database_imbalacced_rate_59.csv"  #ECG_database_imbalacced_rate_more_1_.csv
#Path_sorce_="ECG_database_imbalacced_rate_more_59_.csv"
# ('this is uci data')
#print ('this is Tanzania data')
#print ('ECG data loading')
seg=5 # for uci test or Tan test, the value is 5; for ecg, the value=96
if (os.path.exists(Path_sorce_)):
     ratings = pd.read_csv(Path_sorce_,header=None)
     dataset = ratings.values
     scaler = MinMaxScaler(feature_range=(0, 1))
     dataset = scaler.fit_transform(dataset)
     all_length=len(dataset)
     if(seg==5):
         train_length=(int)(all_length*Per_)
     else: train_length = 120

     trainData=  (dataset[0            : train_length,0:seg]).tolist()
     trainLabels=(dataset[0            : train_length,  seg]).tolist()
     testData=   (dataset[train_length : all_length,  0:seg]).tolist()
     testLabels= (dataset[train_length : all_length,    seg]).tolist()
     t0= time.time()
     #print ("Start")
     ## HD ##
     model = HDFunctions.buildHDModel(trainData, trainLabels, testData, testLabels, D, nLevels)
     #print ("Model generated")
     accuracy = HDFunctions.trainNTimes(model.classHVs, model.trainHVs, model.trainLabels, model.testHVs, model.testLabels, n)

     # training and self-checking
     #accuracy = HDFunctions.trainNTimes(model.classHVs, model.trainHVs, model.trainLabels, model.trainHVs, model.trainLabels, n)
     
     #guess_results = HDFunctions.model_build(model.classHVs, model.trainHVs, model.trainLabels, model.testHVs, model.testLabels, n)    
     #print(str(guess_results))
     
     #prints the maximum accuracy achieved    
     t1 = time.time() - t0
     print("Time elapsed: ", t1 ) # CPU seconds elapsed (floating point)
     print(str(accuracy))
     print('the maximum accuracy is: ' + str(max(accuracy)))
     
else:
    print ("No dataset here")
#current, peak = tracemalloc.get_traced_memory()
#print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
#tracemalloc.stop()
"""     
with open('isolet.pkl', 'rb') as f:
    isolet = pickle.load(f)
trainData, trainLabels, testData, testLabels = isolet
#encodes the training data, testing data, and performs the initial training of the HD model
model = HDFunctions.buildHDModel(trainData, trainLabels, testData, testLabels, D, nLevels)
#retrains the HD model n times and after each retraining iteration evaluates the accuracy of the model with the testing set
accuracy = HDFunctions.trainNTimes(model.classHVs, model.trainHVs, model.trainLabels, model.testHVs, model.testLabels, n)
#prints the maximum accuracy achieved
print('the maximum accuracy is: ' + str(max(accuracy)))
"""