# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 03:55:58 2022

@author: abhis
"""
## write a program to take in a dataset run the poly fit of various degree and test the best fit:

## This class receives the dataset and convert it into objects:
import random
import matplotlib.pyplot as plt
import pylab


def main():
    
    path = r'C:\Users\abhis\OneDrive\Desktop\MIT\6002_Intro_to computational_thinking\Code\Lecture8\temperatures.csv'
    population = readdata(path)
    dict_pop = YearlyTemp(population)
    xval, yval = xval_yval(dict_pop)
    trials = int(input('Enter number of trials: '))
    print(test_model(trials,[1,2,3,4], xval, yval))
    
    
class TempDate(object):
    def __init__(self, s):
        data = s.split(',')
        self.temp = float(data[1])
        self.year = int(data[2][0:4])   ## extract only the year part
        
    def getTemp(self):
        return self.temp
    
    def getYear(self):
        return self.year
              
        
def readdata(path):
    data =[]
    with open(path) as file:
        lines = file.readlines()
        for line in lines[1:]:
            line=line.strip()
            data.append(TempDate(line))
    return data

## get a dictionary of range of temperatures for 

def YearlyTemp(data):
    Year_Temp = {}
    
    for i in data:
        if i.getYear() not in Year_Temp:
            Year_Temp[i.getYear()]=[]
        else:
            Year_Temp[i.getYear()].append(i.getTemp())
    ## calculate the mean of the yearly temperature
    for year in Year_Temp:
        Year_Temp[year] = sum(Year_Temp[year])/len(Year_Temp[year])
    return Year_Temp

def xval_yval(data): ## taeks in the dictionary of the yearly temperature trends
    xval = []
    yval = []
    for year in data:
        xval.append(year)
        yval.append(data[year])
    #plt.plot(xval, yval)
    return xval, yval
## function that distributes the data in testing and training sets:

def Test_Train(xval, yval):
    xtrain, ytrain, xtest, ytest = [], [], [], []
    indices = random.sample(range(len(xval)), len(xval)//2)
    for i in range(len(xval)):
        if i in indices:
            xtrain.append(xval[i])
            ytrain.append(yval[i])
        else:
            xtest.append(xval[i])
            ytest.append(yval[i])
    return xtrain, ytrain, xtest, ytest 

# returns the r squared value 

def rSquared(observed, predicted):
    import numpy
    error = ((predicted - observed)**2).sum()
    meanError = error/len(observed)
    return 1 - (meanError/numpy.var(observed))

## function to run and return the performance of the model:

def test_model(trials, models, xval, yval):    ## exp models =[1,2,3,4]
    ds={}
    for model in models:
        ds[model]=[]
    for i in range(trials):
        xtrain, ytrain, xtest, ytest = Test_Train(xval, yval)
        
        for model in models:
            mdl = pylab.polyfit(xtrain, ytrain, model)
            esty = pylab.polyval(mdl, xtest)
            ds[model].append(rSquared(ytest, esty))
    
    for model in ds:
        ds[model] = sum(ds[model])/len(ds[model])
    return ds   
main()    