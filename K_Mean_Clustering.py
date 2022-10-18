# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:12:00 2022

@author: abhis
"""
def Minkowski(v1, v2, p):
    s= 0
    for i in range(len(v1)):
        s= s+ (v1[i]-v2[i])**p
    return s**(1/p)
        
class Samples(object):
    def __init__(self, name, features, label =None):
        self.name = name
        self.features = features ## these are the dimensions in the sample
        self.label = label
        
    def getName(self):
        return self.name
    
    def getFeature(self):  ## check this one
        return self.features[:]
    
    def getDistance(self, other):
        return Minkowski(self.features, other.getFeature(), 2) # Euclidian
    
    def getDimension(self):
        return len(self.features)
    
    def __str__(self):
        return f"{self.name} has {self.features}"
    
## cluster class
import pylab
class cluster(object):
    def __init__(self, samples):  ## samples are the non empty list of samples of Samples object
        self.samples = samples
        self.centroid = self.computeCentroid()
        
    def update(self, samples):
        oldCentroid = self.centroid
        self.samples = samples
        self.centroid = self.computeCentroid()
        return oldCentroid.getDistance(self.centroid)
    
    def computeCentroid(self):
        vals = pylab.array([0.0]*self.samples[0].getDimension())
        for e in self.samples: #compute mean
            vals += e.getFeature()
        centroid = Samples('centroid', vals/len(self.samples))
        return centroid
    
    def variability(self):
        var = 0
        for i in self.samples:
            var = var + (i.getDistance(self.centroid))**2
        return var
           
    def getCentroid(self):
        return self.centroid
    
    def __str__(self):
        names = []
        for e in self.samples:
            names.append(e.getName())
        names.sort()
        result = 'Cluster with centroid '\
               + str(self.centroid.getFeature()) + ' contains:\n  '
        for e in names:
            result = result + e + ', '
        return result[:-2] #remove trailing comma and space    
        
    
def Dissimilarity(clusters):
    
    Dis = 0
    for c in clusters:
        Dis = Dis +c.variability()
    return Dis
    

def Nor(vals):
    import numpy as np
    vals = np.array(vals)
    mean = sum(vals)/len(vals)
    std = np.std(vals)
    vals = vals - mean
    return vals/std

def getData(toscale = False):
    import numpy as np
    
    hrList, stElevList, ageList, prevACSList, classList = [],[],[],[],[]
    
    with open(r'cardiacData.txt','r') as file: ## add the text file here
        lines = file.readlines()
        for line in lines:
            line = line.strip().split(',')
            hrList.append(int(line[0]))
            stElevList.append(int(line[1]))
            ageList.append(int(line[2]))
            prevACSList.append(int(line[3]))
            classList.append(int(line[4]))
            
        if toscale:
            hrList = Nor(hrList)
            stElevList = Nor(stElevList)
            ageList = Nor(ageList)
            prevACSList = Nor(prevACSList)
        samples = []    
        for i in range(len(hrList)):
            feature = np.array([hrList[i], stElevList[i], ageList[i], prevACSList[i]])
            samples.append(Samples('P'+ str(i), feature, classList[i]))
        return samples
    
    
def Kmeans(samples, k ):  # examples are the list of sample 
    import random
    InitialCentroids = random.sample(samples, k)
    InitialClusters = []
    for i in InitialCentroids:
        InitialClusters.append(cluster([i]))
    
    convergence = False
    while not convergence:
        newcluster = []
        for i in range(k):
            newcluster.append([])
        for i in samples:
            shortdist = i.getDistance(InitialClusters[0].getCentroid())
            index =  0
            for j in range(1,k):
                distance = i.getDistance(InitialClusters[j].getCentroid())
                if distance < shortdist:
                    shortdist = distance
                    index = j
            newcluster[index].append(i)
            
        for c in newcluster:
            if len(c)==0:
                raise ValueError('Empty cluster')
                
        convergence =True
        for i in range(k):
            if InitialClusters[i].update(newcluster[i]) > 0.0:
                convergence = False
                
    return InitialClusters

def testKmean(examples,numclusters, numtrials):
    best = Kmeans(examples, numclusters)
    diss = Dissimilarity(best)
    trial =1
    while trial< numtrials:        ## can work with try:  except
        curr_clus = Kmeans(examples, numclusters)
        curr_diss = Dissimilarity(curr_clus)
        if curr_diss < diss:
            best =curr_clus
            diss = curr_diss
        trial = trial + 1
    return best