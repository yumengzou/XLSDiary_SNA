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
from matplotlib.widgets import Button, CheckButtons
from unittest.mock import inplace

os.chdir("/Users/yumeng.zou/Google Drive/Freshyear/Summer/Research/Diary/")
plt.rcParams['backend']='GTK'
plt.rcParams['font.family']='Microsoft YaHei'

diary=pd.read_csv("csv/Diary.csv")

class ClusterByType():
     
    labels=[]

    def init(self,colName):
        
        ClusterByType.colName=colName
        
        EventCus=diary[['Event',colName]].dropna()
        EventType=diary[['Event','Type']].dropna()
        CusType=pd.merge(EventCus,EventType,how='left').drop("Event",axis=1)  
        CusType['Freq']=1
        ClusterByType.CusType=CusType.groupby([colName,'Type']).sum().reset_index()
        
        ClusterByType.switch=pd.DataFrame({'types':ClusterByType.CusType['Type'].value_counts()[:8].index,
                                           'boo':[False,False,True,True,True,True,True,True]})
        
        filtered=ClusterByType.CusType.query("Type!='Go'&Type!='Visit'")
        ClusterByType.pivot=filtered.pivot(index=colName,columns='Type',values='Freq')
        
        ClusterByType.fig=plt.figure(figsize=(9,6))
          
    def filter(self):
        switch=ClusterByType.switch.set_index('boo')['types']
        thrownFeatures=['Type!='+repr(Type) for Type in switch[False]]
        
        if thrownFeatures:
            s='&'.join(thrownFeatures)
            filtered=ClusterByType.CusType.query(s)
        else:
            filtered=ClusterByType.CusType
        
        ClusterByType.pivot=filtered.pivot(index=ClusterByType.colName,columns='Type',values='Freq')
            
    def draw(self):
        mat=ClusterByType.pivot.replace(np.nan,0).as_matrix()
        clusters=KMeans(n_clusters=10).fit(mat)
        ClusterByType.labels=clusters.labels_
        pca=skl.PCA(n_components=3).fit(mat)
        x,y,z=zip(*pca.transform(mat))
        
        ax = ClusterByType.fig.add_subplot(111, projection='3d')
        points=ax.scatter(y,x,z,c=ClusterByType.labels,s=50,alpha=0.5,picker=3)
        ax.set_title('Clustering Places based on the Type of events they appear in')
        
        # axes labels
        xl,yl,zl=pca.explained_variance_ratio_
        ax.set_xlabel('{:.2f}%'.format(xl*100))
        ax.set_ylabel('{:.2f}%'.format(yl*100))
        ax.set_zlabel('{:.2f}%'.format(zl*100))
        
        # Button: regenerate graph
        axdraw = plt.axes([0.85, 0.03, 0.1, 0.06])
        bdraw=Button(axdraw,'Draw')
        bdraw.on_clicked(ClusterByType.draw)
        
        # Button: close graph and store csv
        axdraw = plt.axes([0.85, 0.9, 0.1, 0.06])
        bclose=Button(axdraw,'Close')
        bclose.on_clicked(ClusterByType.ClosenSave)
        
        # Check Button: filter features
        rax = plt.axes([0.02, 0.3, 0.12, 0.35])
        check = CheckButtons(rax, ClusterByType.switch['types'], ClusterByType.switch['boo'])
        
        def checkFilter(label):
            switch=ClusterByType.switch.set_index('types')['boo']
            switch[label]=not switch.ix[label]
            ClusterByType.switch['boo']=switch.values
            
            ClusterByType.filter(self)
            ClusterByType.draw(self)
            
        check.on_clicked(checkFilter)
        
        # mouse event: print full information of each point
        def onPick(event):
            
            if event.artist != points:
                return True
                
            N = len(event.ind)
            if not N:
                return True
            
            i=0
            for dataidx in event.ind:
                if i==3:
                    break
                print(ClusterByType.pivot.reset_index().loc[dataidx].dropna())
                i+=1
            plt.show()
            return True
        
        ClusterByType.fig.canvas.mpl_connect('pick_event', onPick)
        
        plt.show()
        
    def ClosenSave(self):
        result=pd.DataFrame({'cluster':ClusterByType.labels,'label':ClusterByType.pivot.index}).sort_values(by='cluster',ascending=False)
        result.to_csv("csv/placeCluster.csv",index=False,encoding='utf-8')
        plt.close()



g = ClusterByType()
g.init('Place')
g.draw()


