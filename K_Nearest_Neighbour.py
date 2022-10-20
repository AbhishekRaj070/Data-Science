# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:45:19 2022

@author: abhis
"""
## Titanic dataset to predict people survived
def Minkowski(v1, v2, p):
    s= 0
    for i in range(len(v1)):
        s = s+(v1[i]-v2[i])**p
    return s**(1/p)
        

class Passenger(object):
    
    def __init__(self, pclass, age, gender, label, name):
        self.feature_vec= [0,0,0, age, gender]
        self.feature_vec[pclass -1] = 1
        self.pclass = pclass
        self.label = label
        self.name = name
    
    def getLabel(self):
        return self.label
        
    def getFeature(self):
        return self.feature_vec[:]
        
    def getGender(self):
        return self.feature_vec[4]
    
    def getName(self):
        return self.name
    
    def getAge(self):
        return self.feature_vec[3]
    
    def getDistance(self, other):
        return Minkowski(self.getFeature(), other.getFeature(),2)
    
    def __str__(self):
        return f"{self.name}"
    
    
def getData(path):   ## can use the titanic dataset 
    samples = []                 
    data ={}
    data['pclass'], data['age'], data['gender'], data['label'], data['name']=[],[],[],[],[]
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split(',')
            data['pclass'].append(int(line[0]))
            data['age'].append(float(line[1]))
            if line[2] =='M':
                data['gender'].append(1)
            else:
                data['gender'].append(0)
                
            if int(line[3])==1:
                data['label'].append('survived')
            else:
                data['label'].append('died')
            data['name'].append(line[4:])
        
    # create passenger object and store them in list using Passenger class
    for i in range(len(data['pclass'])):
        samples.append(Passenger(data['pclass'][i], data['age'][i], data['gender'][i], data['label'][i], data['name']))
    
    return samples

def Knearest(sample, samples, k):
    nearest, distance = [], []
    for i in range(k):
        nearest.append(samples[i])
        distance.append(sample.getDistance(samples[i]))
    max_dist = max(distance)
    
    ## now iterate through all the other element to shortlist the k nearest neighbours
    for i in samples[k:]:
        dist = sample.getDistance(i)
        if dist < max_dist:
            index = distance.index(max_dist)
            distance[index] = dist
            nearest[index] = i
            max_dist = max(distance)
    return nearest, distance

## split the training and testing data in 80-20 ratio

def split80_20(samples):
    training, testing =[], []
    import random
    indices = random.sample(range(len(samples)), len(samples)//5)
    
    for i in range(len(samples)):
        if i in indices:
            testing.append(samples[i])
        else:
            training.append(samples[i])
    return training, testing

    
    
def KnnClassifier (training, testing, k, label):
    truep, falsp, trueN, falsN = 0,0,0,0
    for i in testing:
        nearest, distance = Knearest(i, training, k)
        count = 0
        for j in nearest:
            if j.getLabel() == label:
                count = count + 1 
        if count > k//2:
            if i.getLabel() == label :
                truep = truep + 1
            else:
                falsp = falsp + 1
        else:
            if i.getLabel() != label:
                trueN = trueN+1
            else:
                falsN = falsN + 1
    return truep, falsp, trueN, falsN

            
        
    
            
            
        
        
