'''
Created on Jul 25, 2017

@author: yumeng.zou
'''
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import sklearn.decomposition as skl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import CheckButtons

os.chdir("/Users/yumeng.zou/Google Drive/Freshyear/Summer/Research/Diary/")
plt.rcParams['backend']='GTK3Agg'
plt.rcParams['font.family']='Microsoft YaHei'

diary=pd.read_csv("csv/Diary.csv")

def percent(arr):
    arr=[n/arr.sum() for n in arr]
    return arr

class Cluster():
    
    def init(self,sampleCol,featureCol,clusternum=6):
        
        Cluster.sampleCol=sampleCol
        Cluster.featureCol=featureCol
        Cluster.clusternum=clusternum
        
        EventSamp=diary[['Event',sampleCol]].dropna()
        EventFeat=diary[['Event',featureCol]].dropna()
        SampFeat=pd.merge(EventSamp,EventFeat,how='left').drop("Event",axis=1)
        SampFeat['Freq']=1
        SampFeat=SampFeat.groupby([sampleCol,featureCol]).sum().reset_index()
        
        # Sample-Feature-Freq table: select the top frequent instances in the user-selected column
        gt4=SampFeat[sampleCol].value_counts()>1 # boolean Series
        freqSamp=SampFeat[sampleCol].value_counts()[gt4].index
        Cluster.SampFeat=SampFeat.set_index(sampleCol).ix[freqSamp].reset_index()
        Cluster.pivot=Cluster.SampFeat.pivot(index=sampleCol,columns=featureCol,values='Freq').apply(percent,axis=1).replace(np.nan,0)
        
        # 3D scatter plot
        Cluster.figscatter=plt.figure(figsize=(12,6))
        Cluster.axs = Cluster.figscatter.add_subplot(121, projection='3d')
        Cluster.axcheck = Cluster.figscatter.add_axes([0.03, 0.6, 0.1, 0.3]) # left, bottom, width, length
        
        
        # Cluster-Sample table
        Cluster.axt=Cluster.figscatter.add_subplot(122)
            
    # draw 3D scatter plot      
    def draw(self):
        
        mat=Cluster.pivot.replace(np.nan,0).as_matrix()
        cluster=KMeans(n_clusters=Cluster.clusternum).fit(mat)
        
        # clustering result
        Cluster.clusters=pd.DataFrame({Cluster.sampleCol:Cluster.pivot.index,'Cluster':cluster.labels_})
        
        # Cluster--Sample-Feature
        ClSpFt=pd.merge(Cluster.SampFeat,Cluster.clusters,how='left').groupby('Cluster')
        forEachCluster=lambda x : x.pivot(index=Cluster.sampleCol,columns=Cluster.featureCol,values='Freq').apply(percent,axis=1).replace(np.nan,0)
        Cluster.ClSpFt=ClSpFt.apply(forEachCluster)
        Cluster.ClSpFt.to_csv('csv/Cluster'+Cluster.sampleCol+'_on'+Cluster.featureCol+'.csv',encoding='utf-8')
        
        # dimension reduction with PCA
        pca=skl.PCA(n_components=3).fit(mat)
        x,y,z=zip(*pca.transform(mat))
        xyz=pd.DataFrame({'x':x,'y':y,'z':z})
        
        # points by cluster
        Cluster.axs.cla()
        points=[]
        labels=[]
        color=['#50a3ba', '#eac763', '#d94e5d','#4e79a7','#f9713c','#b3e4a1',"#e41a1c",'#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
        
        for i in np.arange(Cluster.clusternum):
            subXYZ=Cluster.clusters.query('Cluster=='+str(i)).join(xyz)
            subPoints=Cluster.axs.scatter(subXYZ['x'],subXYZ['y'],subXYZ['z'],c=color[i],s=50,alpha=0.5,picker=3,visible=True)
            points.append(subPoints)
            labels.append("Cluster " + str(i))
        
        # axes title
        Cluster.axs.set_title('Clustering ' + Cluster.sampleCol +' based on the ' + Cluster.featureCol)
        
        # axes labels
        xl,yl,zl=pca.explained_variance_ratio_
        Cluster.axs.set_xlabel('{:.2f}%'.format(xl*100))
        Cluster.axs.set_ylabel('{:.2f}%'.format(yl*100))
        Cluster.axs.set_zlabel('{:.2f}%'.format(zl*100))
         
        # Check Button: filter clusters
        Cluster.axcheck.cla()
        check = CheckButtons(Cluster.axcheck, labels, [True]*Cluster.clusternum)
          
        def checkCluster(label):
            
            points[labels.index(label)].set_visible(not points[labels.index(label)].get_visible())
            
            plt.draw()
              
        check.on_clicked(checkCluster)
         
        # mouse event: a pie chart for each point on click
        def pickSamp(event):
            
            if event.artist not in points:
                return True
                  
            N = len(event.ind)
            if not N:
                return True
              
            for i,dataidx in enumerate(event.ind):
                if i==3:
                    break
                # pie chart
                figpie,axpie=plt.subplots()
                ser=Cluster.pivot.reset_index().loc[dataidx].dropna()
                Labels=ser[1:].index
                size=[round(n*100/ser[1:].sum()) for n in ser[1:].values]
                Explode=[0.1 if p==max(size) else 0 for p in size]
                axpie.pie(size, explode=Explode, labels=Labels, autopct='%1.1f%%', shadow=True, startangle=90)
                axpie.set_title(ser[0])
              
            plt.show()
            return True
          
        Cluster.figscatter.canvas.mpl_connect('pick_event', pickSamp)
        
        # Sample-Cluster table
        
        Cluster.axt.set_xticks([])
        Cluster.axt.set_yticks([])
        
        meanSamplesperCluster=np.int(Cluster.clusters.groupby('Cluster').count().mean()[Cluster.sampleCol])
        cell_texts=pd.DataFrame("",columns=np.arange(Cluster.clusternum), index=np.arange(meanSamplesperCluster))
        columns=[]
        
        for i, df in Cluster.clusters.groupby('Cluster'):
            replaceLen=min(meanSamplesperCluster,len(df[Cluster.sampleCol]))
            cell_texts.ix[:replaceLen-1,i]=df[Cluster.sampleCol].values[:replaceLen]
            columns.append("Cluster "+str(i))
        
        Cluster.axt.table(cellText=cell_texts.values.tolist(),colLabels=columns,loc='center',fontsize=20)
        
        plt.show()




g1 = Cluster()
g1.init('Place','Type')
g1.draw()

# g2 = Cluster()
# g2.init('Place','Participants')
# g2.draw()

# g3=Cluster()
# g3.init('Participants','Type')
# g3.draw()

# g4=Cluster()
# g4.init('Place','Season')
# g4.draw()
