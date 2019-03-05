#The code is dPtsInside in Python 2.7
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
pathname=os.path.dirname(os.path.abspath(__file__))
	    
plt.style.use('ggplot')
fig,ax=plt.subplots(figsize=(14, 14))
change=dict()
def readFile(a):#read file method
	data=pd.read_csv(pathname+"/"+a)
	X=data.values
	return X
def in_hull(p, hull,pinhull):#whether the point lies inside polygon or not ...(returns how many points lies inside the polygon)
	alreadyptsinhull=len(pinhull)	
	if not isinstance(hull,Delaunay):
		hull = Delaunay(hull)
	asd=( hull.find_simplex(p)>=0)
	b=[x for x in asd if x==True]
	return len(b)
def make_cluster(X):#use DBSCAN to cluster the datasets		
	model=DBSCAN(eps=0.005, min_samples=20).fit(X)
	ax.scatter(X[:,1],X[:,0],c=model.labels_, s=10, alpha=0.9,cmap=plt.cm.Set1)

	labels=model.labels_
	n_clusters_=(len(set(model.labels_)))-(1 if -1 in model.labels_ else 0)
	cluster_pts=dict()
	e=len(X)
	for x in range(n_clusters_):
		for r in range(e):
			if x==labels[r]:
				if x in cluster_pts:
					cluster_pts[x].append((X[r,1],X[r,0]))
				else:
					cluster_pts[x]=[(X[r,1],X[r,0])]
	for x in cluster_pts:
		points=np.array(cluster_pts[x])
		points1=np.array(cluster_pts[1])
		hull=ConvexHull(points)
		for simplex in hull.simplices:
			ax.plot(points[simplex, 0], points[simplex, 1], 'k-')
	plt.show()	
	return cluster_pts

def clusterProperty(PtsInside,ptsTwo,ptsThree):# method to find whether the cluster shrink, expand or dissappear
	print(PtsInside,ptsTwo,ptsThree)	
	if ((float(PtsInside)/ptsThree))>0.80:
		if ptsTwo>ptsThree:
			return 1
		else:
			return -1
	else:
		return 0
if __name__=="__main__":
		
	st=["April.csv","June.csv"]#no. of dataset	
	for t in range(len(st)):	
		if t==0:
			X=readFile(st[t])					
			cluster_pts1=make_cluster(X)		
			formation=len(cluster_pts1)			
			change[st[t]]=[formation]				
		else:		
			X=readFile(st[t-1])
			Y=readFile(st[t])	
	
			cluster_pts1=make_cluster(X)
			cluster_pts2=make_cluster(Y)	
	
			changes=[]	
			for y in cluster_pts1:		
				for x in cluster_pts2:	
				
					points2=np.array(cluster_pts2[x])
					points1=np.array(cluster_pts1[y])#cluster 1 points of 1 dataset		
					hull=ConvexHull(points2)#all hull of 2 dataset 		
				
			
					PtsInside=in_hull(points1,points2[hull.vertices],points2)		
					ptsTwo=len(points2)
					ptsThree=len(points1)
					#print("points of t0 appears in t1  hull",PtsInside)  
			
					#print("points of t1 surrounded in t1 hull ",ptsTwo)
					#print("points of t0",ptsThree)
		
		
					pr= clusterProperty(PtsInside,ptsTwo,ptsThree)
					if pr!=0:
					
						changes.append(pr)		
						break		
					#print("******")
			dis=len(cluster_pts1)-len(changes)
			formation=len(cluster_pts2)-len(changes)	
			expansion=len([i for i in changes if i==1])	
			shrink=len([i for i in changes if i==-1])	
			change[st[t]]=[expansion,shrink,dis,formation]
	print(change)


