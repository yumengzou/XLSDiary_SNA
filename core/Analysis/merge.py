'''
Created on Jul 25, 2017

@author: yumeng.zou
'''
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.chdir("/Users/yumeng.zou/Google Drive/Freshyear/Summer/Research/Diary/")

# make place-type table from Diary
diary=pd.read_csv("csv/Diary.csv")
evt_plc=diary[['Event','Place']].dropna()
evt_type=diary[['Event','Type']].dropna()
plc_type=pd.merge(evt_plc,evt_type,how='left').drop("Event",axis=1)

# count the frequency of types for each place
plc_type['Freq']=1
plc_type=plc_type.groupby(['Place','Type']).sum().reset_index()#.sort_values(by='Freq',ascending=False)
### remove 'Go' from the clustering process because 'Go' is very frequent but does not carry cultural meanings
plc_type=plc_type.query("Type!='Go'&Type!='Visit'") 

# make sample(place)-feature(type) table from place, type, and freq
pivot=plc_type.pivot(index='Place',columns='Type',values='Freq')
mat=pivot.replace(np.nan,0).as_matrix()

# cluster places with K-means
cluster=KMeans(n_clusters=6).fit(mat)

# visualize clusters with PCA(principle component analysis)
pca=PCA(n_components=3).fit(mat)
# print(pca.explained_variance_ratio_)
x,y,z=zip(*pca.transform(mat))
 
fig=plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y,z,x,c=cluster.labels_,s=100,alpha=0.5)
 
plt.savefig("Graph/placeType.png")
plt.close()
 
result=pd.DataFrame({'cluster':cluster.labels_,'label':pivot.index}).sort_values(by='cluster',ascending=False)
result.to_csv("csv/placeCluster.csv",index=False,encoding='utf-8')

