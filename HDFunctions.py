#!/usr/local/bin python
from __future__ import division
import os
import sys
import os.path
import struct
import numpy as np
import math
import copy
from sklearn.metrics import accuracy_score, precision_recall_curve,f1_score
from matplotlib import pyplot as plt
from numpy import linalg as li
import random
import pickle
from math import log, ceil, floor
import warnings
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.metrics import roc_curve, auc
levelHV_bank=[]
training_key_bank=[]
warnings.filterwarnings("ignore")

baseVal = -1


class HDModel(object):
    # Initializes a HDModel object
    # Inputs:
    #   trainData: training data
    #   trainLabels: training labels
    #   testData: testing data
    #   testLabels: testing labels
    #   D: dimensionality
    #   totalLevel: number of level hypervectors
    # Outputs:
    #   HDModel object
    def __init__(self, trainData, trainLabels, testData, testLabels, D, totalLevel):
        if len(trainData) != len(trainLabels):
            print("Training data and training labels are not the same size")
            return
        if len(testData) != len(testLabels):
            print("Testing data and testing labels are not the same size")
            return
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.testData = testData
        self.testLabels = testLabels
        self.D = D
        self.totalLevel = totalLevel
        self.levelList = getlevelList(self.trainData, self.totalLevel)
        self.levelHVs = genLevelHVs(self.totalLevel, self.D)
        self.trainHVs = []
        self.testHVs = []
        self.classHVs = []

    # Encodes the training or testing data into hypervectors and saves them or
    # loads the encoded traing or testing data that was saved previously
    # Inputs:
    #   mode: decided to use train data or test data
    #   D: dimensionality
    #   dataset: name of the dataset
    # Outputs:
    #   none
    def buildBufferHVs(self, mode, D):
        if mode == "train":
            # print("Encoding Training Data")
            # print (len(self.trainData))
            for index in range(len(self.trainData)):
                self.trainHVs.append(EncodeToHV(np.array(self.trainData[index]), self.D, self.levelHVs, self.levelList))
            with open('train_bufferHVs_' + str(D) + '.pkl', 'wb') as f:
                pickle.dump(self.trainHVs, f)
            self.classHVs = oneHvPerClass(self.trainLabels, self.trainHVs, self.D)
            # print("Training Data Encoded")
            # print (self.classHVs)
        else:
            # print("Encoding Testing Data")
            # print (len(self.testData))
            for index in range(len(self.testData)):
                # print (index)
                self.testHVs.append(EncodeToHV(np.array(self.testData[index]), self.D, self.levelHVs, self.levelList))
            # print("Testing Data Enchoded, try to save")
            with open('test_bufferHVs_' + str(D) + '.pkl', 'wb') as f:
                pickle.dump(self.testHVs, f)
            # print("Testing Data Enchoded")
            # print (self.testHVs)


# Performs the initial training of the HD model by adding up all the training
# hypervectors that belong to each class to create each class hypervector
# Inputs:
#   inputLabels: training labels
#   inputHVs: encoded training data
#   D: dimensionality
# Outputs:
#   classHVs: class hypervectors
def oneHvPerClass(inputLabels, inputHVs, D):
    # This creates a dict with no duplicates
    classHVs = dict()
    for i in range(len(inputLabels)):
        name = inputLabels[i]
        if (name in classHVs.keys()):
            classHVs[name] = np.array(classHVs[name]) + np.array(inputHVs[i])
        else:
            classHVs[name] = np.array(inputHVs[i])
    return classHVs

def inner_product(x, y):
    return np.dot(x, y) / (li.norm(x) * li.norm(y) + 0.0)

# Finds the level hypervector index for the corresponding feature value
# Inputs:
#   value: feature value
#   levelList: list of level hypervector ranges
# Outputs:
#   keyIndex: index of the level hypervector in levelHVs corresponding the the input value
def numToKey(value, levelList):
    if (value == levelList[-1]):
        return len(levelList) - 2
    upperIndex = len(levelList) - 1
    lowerIndex = 0
    keyIndex = 0
    while (upperIndex > lowerIndex):
        keyIndex = int((upperIndex + lowerIndex) / 2)
        if (levelList[keyIndex] <= value and levelList[keyIndex + 1] > value):
            return keyIndex
        if (levelList[keyIndex] > value):
            upperIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex) / 2)
        else:
            lowerIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex) / 2)
    # print ("keyIndex")
    # print (keyIndex)
    return keyIndex


# Splits up the feature value range into level hypervector ranges
# Inputs:
#   buffers: data matrix
#   totalLevel: number of level hypervector ranges
# Outputs:
#   levelList: list of the level hypervector ranges
def getlevelList(buffers, totalLevel):
    minimum = buffers[0][0]
    maximum = buffers[0][0]
    levelList = []
    for buffer in buffers:
        localMin = min(buffer)
        localMax = max(buffer)
        if (localMin < minimum):
            minimum = localMin
        if (localMax > maximum):
            maximum = localMax
    length = maximum - minimum
    gap = length / totalLevel
    for lv in range(totalLevel):
        levelList.append(minimum + lv * gap)
        # print(minimum + lv*gap)
    levelList.append(maximum)
    # print("Level list generated")
    # print (levelList)
    return levelList


# Generates the level hypervector dictionary
# Inputs:
#   totalLevel: number of level hypervectors
#   D: dimensionality
# Outputs:
#   levelHVs: level hypervector dictionary
def genLevelHVs(totalLevel, D):
    # print ('generating level HVs')
    levelHVs = dict()
    indexVector = range(D)
    print("indexVector",indexVector)
    nextLevel = int((D / 2 / totalLevel))
    change = int(D / 2)
    print("nextLevel", (indexVector)[:nextLevel])
    print("change", (indexVector)[:change])
    fig, (ax0, ax00) = plt.subplots(2, 1)
    for level in range(totalLevel):
        name = level
        if (level == 0):
            base = np.full(D, baseVal)  # 10.000di--> [-1,-1,...,-1]
            toOne = np.random.permutation(indexVector)[:change] #随机产生 50个数， 变化范围都是 0-100
        else:
            toOne = np.random.permutation(indexVector)[:nextLevel] #随机产生12个数
        for index in toOne:
            base[index] = base[index] * -1
        levelHVs[name] = copy.deepcopy(base)
        #ax00.plot(base)
        levelHV_stash('1013', levelHVs[name])#establish a stash bank for initial data saving
    # print ("Level_HV dict generated")
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1)
    ax1.plot(levelHVs[0],  linewidth=1)
    ax2.plot(levelHVs[1], linewidth=1)

    ax3.plot(levelHVs[2],  linewidth=1)
    ax4.plot(levelHVs[3],  linewidth=1)

    plt.xlabel('Example (Dimension=100)')

    #plt.show()
    return levelHVs

def levelHV_stash(password,vectors):
    if(password=='1013'):
        levelHV_bank.append(vectors)

# Encodes a single datapoint into a hypervector
# Inputs:
#   inputBuffer: data to encode
#   D: dimensionality
#   levelHVs: level hypervector dictionary
#   IDHVs: ID hypervector dictionary
# Outputs:
#   sumHV: encoded data
def EncodeToHV(inputBuffer, D, levelHVs, levelList):
    sumHV = np.zeros(D, dtype=np.int)
    #print(len(inputBuffer))==5
    order=[]
    for keyVal in range(len(inputBuffer)):
        key = numToKey(inputBuffer[keyVal], levelList)
        order.append(key)
        # print ("kkk")
        levelHV = levelHVs[key]
        #print (levelHV)#len=10.000
        sumHV = sumHV + np.roll(levelHV, keyVal)
    training_key_stash('1013',order)
    return sumHV

def training_key_stash(password,order):
    if(password=='1013'):
        training_key_bank.append(order)
# This function attempts to guess the class of the input vector based on the model given
# Inputs:
#   classHVs: class hypervectors
#   inputHV: query hypervector
# Outputs:
#   guess: class that the model classifies the query hypervector as
def checkVector(classHVs, inputHV):
    guess = list(classHVs.keys())[0]
    # print (classHVs)
    maximum = np.NINF
    count = {}
    for key in classHVs.keys():
        count[key] = inner_product(classHVs[key], inputHV)
        if (count[key] > maximum):
            guess = key
            maximum = count[key]
    return guess, count


# Iterates through the training set once to retrain the model
# Inputs:
#   classHVs: class hypervectors
#   testHVs: encoded train data
#   testLabels: training labels
# Outputs:
#   retClassHVs: retrained class hypervectors
#   error: retraining error rate
def trainOneTime(classHVs, trainHVs, trainLabels):
    retClassHVs = copy.deepcopy(classHVs)
    wrong_num = 0
    for index in range(len(trainLabels)):
        guess, dis = checkVector(retClassHVs, trainHVs[index])
        if not (trainLabels[index] == guess):
            wrong_num += 1
            retClassHVs[guess] = retClassHVs[guess] - trainHVs[index]
            retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + 1*trainHVs[index]
    error = (wrong_num + 0.0) / len(trainLabels)
    return retClassHVs, error


def trainOneTime_imbalanced(classHVs, trainHVs, trainLabels):
    retClassHVs = copy.deepcopy(classHVs)
    #fig, (ax1, ax2) = plt.subplots(2, 1)
    #ax1.plot(classHVs[0], "y", label='hybird', linewidth=1)
    #ax2.plot(classHVs[1], "y", label='hybird', linewidth=1)
    wrong_num = 0
    neg=0
    plus=0
    for index in range(len(trainLabels)):
        guess, dis = checkVector(retClassHVs, trainHVs[index])
        if not (trainLabels[index] == guess):
            wrong_num += 1
            if(guess==1): neg+=1
            else: plus+=1
            Dimen=len(levelHV_bank)
            re_training_sample=copy.deepcopy(trainHVs[index])
            re_training_sample_new, done=remodel (Dimen,re_training_sample,classHVs,guess,trainLabels,index)
            #retClassHVs[guess] = retClassHVs[guess] - trainHVs[index]
            #retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + trainHVs[index]
            if (done == 1):
                retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + re_training_sample_new
                retClassHVs[guess] = retClassHVs[guess] -re_training_sample_new
            else:
                retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + re_training_sample
                if(trainLabels[index] == 1):
                   retClassHVs[guess] = retClassHVs[guess] - re_training_sample
                #retClassHVs[guess] = retClassHVs[guess] - re_training_sample_new
            #retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + re_training_sample_new
            '''
            if ((done==1)&(trainLabels[index]==1)):
               retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + re_training_sample_new
               print('1')
            elif((done == 0) & (trainLabels[index] == 1)):
                retClassHVs[guess] = retClassHVs[guess] - re_training_sample_new
                retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + re_training_sample_new
                print('2')
            else: retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + re_training_sample_new
            '''
    error=0
    return retClassHVs, error

def remodel (Dimen,re_training_sample,classHVs, guess,trainLabels,index):
    done=0
    key_features=training_key_bank[index]
    #for x in range (Dimen):
    for x in key_features:
        re_training_sample_x = re_training_sample + levelHV_bank[x]
        count_0 = inner_product(classHVs[guess], re_training_sample_x)
        count_1 = inner_product(classHVs[trainLabels[index]], re_training_sample_x)
        if ((count_0 - count_1) < 0.000):
            re_training_sample = re_training_sample_x
            #print("remodeled")
            done = 1
            break
    return (re_training_sample, done)


# Guess--> HD model on the testing set
# Inputs:
#   classHVs: class hypervectors
#   testHVs: encoded test data
#   testLabels: testing labels
# Outputs:
#   accuracy: test accuracy
def Guess_(classHVs, testHVs, testLabels):
    for index in range(len(testHVs)):
        guess, dis = checkVector(classHVs, testHVs[index])
    return (guess)


# Tests the HD model on the testing set
# Inputs:
#   classHVs: class hypervectors
#   testHVs: encoded test data
#   testLabels: testing labels
# Outputs:
#   accuracy: test accuracy
def test(classHVs, testHVs, testLabels):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    sensitivity = 0
    Specificity = 0
    #f_ = open("HYvectors.csv", "w+")
    #f_.truncate()
    #s_ = str(classHVs[0]) + '\n' +str(classHVs[1])
    #s_=''
    #for i in classHVs[0]:
        #s_+=str(i)+'/'
    #f_.write(s_.replace('[', '').replace(']', ''))
    #f_.close()
    #fig, (ax1, ax2) = plt.subplots(2, 1)
    #ax1.plot(classHVs[0], "y", label='hybird', linewidth=1)
    #ax2.plot(classHVs[1], "y", label='hybird', linewidth=1)
    #plt.show()
    testPredict_ = np.zeros(len(testHVs))
    #print("test length: ", len(testHVs))
    for index in range(len(testHVs)):
        guess, dis = checkVector(classHVs, testHVs[index])
        #print("guess_distance ",dis[guess])
        #print("opposite_distance ", dis[1-guess])
        testPredict_[index] = guess
        if (guess == 1):
            # ones_num+=1
            if (testLabels[index] == 1):  # 猜是错的，实际也是错的
                TN += 1
            else:  # 猜是错的，实际是对的
                FN += 1
        elif (guess == 0):
            if (testLabels[index] == 0):  # 猜是对的，实际也是对的
                TP += 1
            else:  # 猜是对的，实际是错的
                FP += 1

    if ((TP != 0) or (FN != 0)):
        sensitivity = TP / (TP + FN)

    else:
        sensitivity = 0
    #print("Sensitivity: " + str(sensitivity))
    if ((FP != 0) or (TN != 0)):
        Specificity = TN / (FP + TN)

    else:
        Specificity = 0
    print("TP " + str(TP))
    print("FP " + str(FP))
    print("FN " + str(FN))
    print("TN " + str(TN))
    time_ = sensitivity * Specificity
    G_Mean = math.sqrt(time_)
    MCC=((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    accuracy_ = (TP+TN)/(TP+TN+FP+FN)
    precision = TP / (TP + FP)
    recall = sensitivity  # TP/(TP+FN)
    f_1= 2 * (precision * recall) / (precision + recall)
    #f_1 = f1_score(testLabels, testPredict_, average='micro')

    #'''
    print('The sensitivity is: ' + str(sensitivity))
    print('The Specificity is: ' + str(Specificity))
    print("MCC: " + str(MCC))
    print("G_Mean: " + str(G_Mean))
    print("accuracy_ is :", accuracy_)
    print("F 1 score is :", f_1)
    print("precision is :", precision)
    print("recall is :", recall)
    #'''


    return (G_Mean)


"""            
        if (testLabels[index] == guess):
            correct += 1
         accuracy = (correct / len(testLabels)) * 100
         accuracy=correct_ones/ones_num
   #print ('The classification accuracy is: ' + str(accuracy))
    return (accuracy)
"""


# Retrains the HD model n times and evaluates the accuracy of the model
# after each retraining iteration
# Inputs:
#   classHVs: class hypervectors
#   trainHVs: encoded training data
#   trainLabels: training labels
#   testHVs: encoded test data
#   testLabels: testing labels
# Outputs:
#   accuracy: array containing the accuracies after each retraining iteration
def trainNTimes(classHVs, trainHVs, trainLabels, testHVs, testLabels, n):
    accuracy = []
    currClassHV = copy.deepcopy(classHVs)
    accuracy.append(test(currClassHV, testHVs, testLabels))
    for i in range(n):
        #currClassHV, error = trainOneTime(currClassHV, trainHVs, trainLabels)
        currClassHV, error = trainOneTime_imbalanced(currClassHV, trainHVs, trainLabels)
        #print ("error",error)
        #if(error<=0.01): break
        accuracy.append(test(currClassHV, testHVs, testLabels))
    return accuracy


# model_building
# after each retraining iteration
# Inputs:
#   classHVs: class hypervectors
#   trainHVs: encoded training data
#   trainLabels: training labels
#   testHVs: encoded test data
#   testLabels: testing labels
# Outputs:
#   accuracy: array containing the accuracies after each retraining iteration
def model_build(classHVs, trainHVs, trainLabels, testHVs, testLabels, n):
    results = []
    currClassHV = copy.deepcopy(classHVs)
    # results.append(Guess_(currClassHV, testHVs, testLabels))
    for i in range(n):
        results.append(test(currClassHV, testHVs, testLabels))
    return results


# Creates an HD model object, encodes the training and testing data, and
# performs the initial training of the HD model
# Inputs:
#   trainData: training set
#   trainLabes: training labels
#   testData: testing set
#   testLabels: testing labels
#   D: dimensionality
#   nLevels: number of level hypervectors
#   datasetName: name of the dataset
# Outputs:
#   model: HDModel object containing the encoded data, labels, and class HVs
def buildHDModel(trainData, trainLabels, testData, testLables, D, nLevels):
    model = HDModel(trainData, trainLabels, testData, testLables, D, nLevels)
    # model.buildBufferHVs("test", D)
    model.buildBufferHVs("train", D)
    model.buildBufferHVs("test", D)
    return model

