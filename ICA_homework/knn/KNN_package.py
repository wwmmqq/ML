#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 21:59:00 2014

@author: wepon

@blog:http://blog.csdn.net/u012162613
"""
import numpy

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from numpy import *

import csv
import os


def toFloat(array):
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i, j] = float(array[i, j])
    return newArray


def nomalizing(array):
    m, n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array


'''
train.csv是训练样本集，大小42001*785，第一行是文字描述，所以实际的样本数据大小是42000*785，
其中第一列的每一个数字是它对应行的label，可以将第一列单独取出来，得到42000*1的向量trainLabel，
剩下的就是42000*784的特征向量集trainData，所以从train.csv可以获取两个矩阵trainLabel、trainData。
'''


def loadTrainData(trainPath):
    l = []
    with open(trainPath) as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
            print(line)
            b = array(l)
    # l.remove(l[0])
    l = array(l)
    # l = numpy.matrix(l)
    label = l[:, 0]
    data = l[:, 1:]
    return nomalizing(toFloat(data)), toFloat(label)  # label 1*42000  data 42000*784
    # return trainData,trainLabel


def loadTestData(testPath):
    l = []
    with open(testPath) as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 28001*784
    # l.remove(l[0])
    l = array(l)
    label = l[:, 0]
    data = l[:, 1:]
    return nomalizing(toFloat(data)), toFloat(label)  # data 28000*784
    # return testData


'''
def loadTestResult():
    l = []
    with open('knn_benchmark.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 28001*2
    l.remove(l[0])
    label = array(l)
    return toInt(label[:, 1])  # label 28000*1
'''


# result是结果列表
# csvName是存放结果的csv文件名
def saveResult(result, csvName):
    with open(csvName, 'wb') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)


# 调用scikit的knn算法包
def knnClassify(trainData, trainLabel, testData):
    knnClf = KNeighborsClassifier()  # default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData, ravel(trainLabel))
    testLabel = knnClf.predict(testData)
    saveResult(testLabel, 'sklearn_knn_Result.csv')
    return testLabel


# 调用scikit的SVM算法包

def svcClassify(trainData, trainLabel, testData):
    svcClf = svm.SVC(
        C=5.0)  # default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    svcClf.fit(trainData, ravel(trainLabel))
    testLabel = svcClf.predict(testData)
    saveResult(testLabel, 'sklearn_SVC_C=5.0_Result.csv')
    return testLabel


# 调用scikit的朴素贝叶斯算法包,GaussianNB和MultinomialNB
# nb for 高斯分布的数据
def GaussianNBClassify(trainData, trainLabel, testData):
    nbClf = GaussianNB()
    nbClf.fit(trainData, ravel(trainLabel))
    testLabel = nbClf.predict(testData)
    saveResult(testLabel, 'sklearn_GaussianNB_Result.csv')
    return testLabel

    # nb for 多项式分布的数据


def MultinomialNBClassify(trainData, trainLabel, testData):
    nbClf = MultinomialNB(
        alpha=0.1)  # default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing.
    nbClf.fit(trainData, ravel(trainLabel))
    testLabel = nbClf.predict(testData)
    saveResult(testLabel, 'sklearn_MultinomialNB_alpha=0.1_Result.csv')
    return testLabel


# RandomForestClassifier
def RandomFClassifier(trainData, trainLabel, testData):
    nbClf = RandomForestClassifier()
    nbClf.fit(trainData, ravel(trainLabel))
    testLabel = nbClf.predict(testData)
    saveResult(testLabel, 'sklearn_randomforestclassifier_n_estimators=10_Result.csv')
    return testLabel


def digitRecognition(trainPath, testPath):
    trainData, trainLabel = loadTrainData(trainPath)
    testData, testLabel = loadTestData(testPath)
    # 使用不同算法
    # result = RandomFClassifier(trainData, trainLabel, testData)
    result = knnClassify(trainData, trainLabel, testData)
    # result2 = svcClassify(trainData, trainLabel, testData)
    # result3 = GaussianNBClassify(trainData, trainLabel, testData)
    # result4 = MultinomialNBClassify(trainData, trainLabel, testData)
    print(result)
    # 将结果与跟给定的knn_benchmark对比,以result1为例
    m, n = shape(testData)
    different = 0.0  # result1中与benchmark不同的label个数，初始化为0
    for i in xrange(m):
        if result[i] != testLabel[0, i]:
            different += 1.0
    rate = different / m
    file = open('D:\\RiousSvn_Code\\LDAModel_GibbsSampling\\rate.txt', 'w')
    file.write(trainPath + '+' + testPath + ':' + str(rate * 100))
    file.close()


def CaculateEachParme():
    # csvDir = 'D:\\RiousSvn_Code\\LDAModel_GibbsSampling'
    trainPath = 'D:\\RiousSvn_Code\\LDAModel_GibbsSampling\\generatedTrain.csv'
    # trainPath = 'D:\\RiousSvn_Code\\LDAModel_GibbsSampling\\test.csv'
    testPath = 'D:\\RiousSvn_Code\\LDAModel_GibbsSampling\\generatedTest.csv'
    digitRecognition(trainPath, testPath)
    # fileDir = os.listdir(csvDir)


'''
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='1and75%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='1and75%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='3and75%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='3and75%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='4and75%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='4and75%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='5and75%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='5and75%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='6and75%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='6and75%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='extend35%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='extend35%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='extend45%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='extend45%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='extend55%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='extend55%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='extend65%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='extend65%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='extend75%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='extend75%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='extend85%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='extend85%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='notextend35%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='notextend35%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='notextend45%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='notextend45%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='notextend55%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='notextend55%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='notextend65%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='notextend65%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='notextend75%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='notextend75%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)
    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if(cateName=='notextend85%' and fileName=='generatedTest.csv'):
            testPath = csvDir+'\\'+filePath
        elif(cateName=='notextend85%' and fileName=='generatedTrain.csv'):
            trainPath = csvDir+'\\'+filePath
    digitRecognition(trainPath,testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if (cateName == 'computerNewdatanoword2vec' and fileName == 'generatedTest.csv'):
            testPath = csvDir + '\\' + filePath
        elif (cateName == 'computerNewdatanoword2vec' and fileName == 'generatedTrain.csv'):
            trainPath = csvDir + '\\' + filePath
    digitRecognition(trainPath, testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if (cateName == 'computerOlddataword2vec' and fileName == 'generatedTest.csv'):
            testPath = csvDir + '\\' + filePath
        elif (cateName == 'computerOlddataword2vec' and fileName == 'generatedTrain.csv'):
            trainPath = csvDir + '\\' + filePath
    digitRecognition(trainPath, testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if (cateName == 'medicalNewdatanoWordvec' and fileName == 'generatedTest.csv'):
            testPath = csvDir + '\\' + filePath
        elif (cateName == 'medicalNewdatanoWordvec' and fileName == 'generatedTrain.csv'):
            trainPath = csvDir + '\\' + filePath
    digitRecognition(trainPath, testPath)

    for filePath in fileDir:
        cateName = filePath.split('--')[0]
        fileName = filePath.split('--')[1]
        if (cateName == 'medicalOlddatanoWordvec' and fileName == 'generatedTest.csv'):
            testPath = csvDir + '\\' + filePath
        elif (cateName == 'medicalOlddatanoWordvec' and fileName == 'generatedTrain.csv'):
            trainPath = csvDir + '\\' + filePath
    digitRecognition(trainPath, testPath)
'''

if __name__ == '__main__':
    CaculateEachParme()
