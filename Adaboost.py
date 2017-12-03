# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:04:34 2017

@author: Q
"""
import numpy as np
import matplotlib.pyplot as plt
def loadSimple():   #创建训练样本
    dataMat = np.mat([[1,2.1],
                      [2,1.1],
                      [1.3,1],
                      [1,1],
                      [2,1]])
    classLabels = [1,1,-1,-1,1]
    return dataMat,classLabels
def strumpClassify(dataMatrix,dimen,threshVal,threshIneq):  #分类器分类
    retArray = np.ones((dataMatrix.shape[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1
    return retArray

def buildStrump(dataMatrix,classLabels,D):    #选择最优分类器
    dataMat = np.mat(dataMatrix)
    classMat = np.mat(classLabels).T
    m,n = np.shape(dataMat)
    numSteps = 10
    minError = np.inf
    bestStrump = {}
    bestClassEst = np.zeros((m,1))
    for dim in range(n):
        minlim = float(dataMat[:,dim].min())
        maxlim = float(dataMat[:,dim].max())
        stepLength = (maxlim - minlim)/numSteps
        for step in range(-1,int(numSteps)+1):
            threshValue = minlim + step * stepLength
            for inequal in ['lt','gt']:
                predictArray = np.mat(strumpClassify(dataMat,dim,threshValue,inequal))
                errorArray = np.ones((m,1))
                errorArray[predictArray == classMat] = 0
                weightError = D.T*errorArray
                if weightError < minError:
                    minError = weightError
                    bestStrump['dim'] = dim
                    bestStrump['thresh'] = threshValue
                    bestStrump['ineq'] = inequal
                    bestClassEst = predictArray.copy()
    return bestStrump,minError,bestClassEst

    
def adaBoostTrainDS(dataArr,classLabels,numIt = 40):#训练得到AdaBoost分类器
    weekClassArr = []
    m = np.shape(dataArr)[0]
    dataMat = np.mat(dataArr)
    aggClassEst = np.mat(np.zeros((m,1)))
    D = np.mat(np.ones((m,1))/m)
    for i in range(numIt):
        nowStump,nowError,nowClassEst = buildStrump(dataMat,classLabels,D)
        nowalpha = float(0.5*np.log((1-nowError)/max(nowError,1e-16)))
        nowStump['alpha'] = nowalpha
        weekClassArr.append(nowStump)
        expon = np.multiply(-1*nowalpha*np.mat(classLabels).T,nowClassEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        aggClassEst = aggClassEst + nowalpha * nowClassEst
        aggError = np.mat(np.zeros((m,1)))
        aggError[np.sign(aggClassEst) != np.mat(classLabels).T] = 1
        errorRate = aggError.sum()/m
        print('iter:%d   errorRate:'%i,errorRate)
        if errorRate == 0:
            break
    return weekClassArr
    
def adaClassify(dataToClass,classifierArr):   #使用分类器分类
    dataMatrix = np.mat(dataToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for nowClassifier in classifierArr:
        classEst = strumpClassify(dataMatrix,nowClassifier['dim'],nowClassifier['thresh'],nowClassifier['ineq'])
        aggClassEst = aggClassEst + nowClassifier['alpha']*classEst
    return aggClassEst
    
D = np.mat(np.ones((5,1))/5)
data,classLabels = loadSimple()
weekClassifiers = adaBoostTrainDS(data,classLabels,9)


fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.scatter(data[:,0],data[:,1],c = classLabels,s=30, cmap=plt.cm.Paired)

xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
xx = np.linspace(xmin,xmax,30)
yy = np.linspace(ymin,ymax,30)
xxx,yyy = np.meshgrid(xx,yy)
xy = np.dstack((xxx,yyy)).reshape(np.size(xxx),2)
xyLabels = adaClassify(xy,weekClassifiers).reshape(xxx.shape)
ax.contour(xxx,yyy,xyLabels,levels=[-1,0,1],colors = 'b',linestyles=['--','-','-- '])


plt.show()



