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

os.chdir("/Users/yumeng.zou/Google Drive/Freshyear/Summer/Research/Diary/")
plt.rcParams['backend']='GTK'
plt.rcParams['font.family']='Microsoft YaHei'

diary=pd.read_csv("csv/Diary.csv")

class byType():
    
    def init(self,colName):
        
        byType.colName=colName
        
        EventCus=diary[['Event',colName]].dropna()
        EventType=diary[['Event','Type']].dropna()
        CusType=pd.merge(EventCus,EventType,how='left').drop("Event",axis=1)  
        CusType['Freq']=1
        byType.CusType=CusType.groupby([colName,'Type']).sum().reset_index()
        
        byType.switch=pd.DataFrame({'types':byType.CusType['Type'].value_counts()[:6].index,
                                    'boo':[False,False,True,True,True,True]})
        
        filtered=byType.CusType.query("Type!='Go'&Type!='Visit'")
        byType.pivot=filtered.pivot(index=colName,columns='Type',values='Freq')
        
        byType.figscatter=plt.figure(figsize=(9,6))
        byType.axs = byType.figscatter.add_subplot(111, projection='3d')
        byType.figbar=plt.figure(figsize=(9,6))
        byType.axb=byType.figbar.add_subplot(111)
          
    def filter(self):
        switch=byType.switch.set_index('boo')['types']
        thrownFeatures=['Type!='+repr(Type) for Type in switch[False]]
        
        if thrownFeatures:
            s='&'.join(thrownFeatures)
            filtered=byType.CusType.query(s)
        else:
            filtered=byType.CusType
        
        byType.pivot=filtered.pivot(index=byType.colName,columns='Type',values='Freq')
            
    def drawScatter(self):
        
        # 3D scatter plot
        mat=byType.pivot.replace(np.nan,0).as_matrix()
        cluster=KMeans(n_clusters=10).fit(mat)
        byType.labels=cluster.labels_
        byType.clusters=pd.DataFrame({byType.colName:byType.pivot.index,'Cluster':byType.labels})
        byType.ClbyTp=pd.merge(byType.clusters,byType.CusType,how='left')\
        .groupby(['Cluster','Type']).count()[byType.colName]\
        .unstack().replace(np.nan,0) # for bar plot
        
        pca=skl.PCA(n_components=3).fit(mat)
        x,y,z=zip(*pca.transform(mat))
        
        byType.axs.cla()
        points=byType.axs.scatter(y,x,z,c=byType.labels,s=50,alpha=0.5,picker=3)
        byType.axs.set_title('Clustering Places based on the Type of events they appear in')
        
        # axes labels
        xl,yl,zl=pca.explained_variance_ratio_
        byType.axs.set_xlabel('{:.2f}%'.format(xl*100))
        byType.axs.set_ylabel('{:.2f}%'.format(yl*100))
        byType.axs.set_zlabel('{:.2f}%'.format(zl*100))
        
        # Button: regenerate graph
        axdraw = byType.figscatter.add_axes([0.85, 0.03, 0.1, 0.06])
        bdraw=Button(axdraw,'Draw')
        bdraw.on_clicked(byType.drawScatter)
        
        # Button: close graph and store csv
        axclose = byType.figscatter.add_axes([0.85, 0.9, 0.1, 0.06])
        bclose=Button(axclose,'Close')
        bclose.on_clicked(byType.ClosenSave)
        
        # Check Button: filter features
        axcheck = byType.figscatter.add_axes([0.02, 0.3, 0.12, 0.35])
        check = CheckButtons(axcheck, byType.switch['types'], byType.switch['boo'])
        
        def checkFilter(label):
            switch=byType.switch.set_index('types')['boo']
            switch[label]=not switch.ix[label]
            byType.switch['boo']=switch.values
            
            byType.filter(self)
            byType.drawScatter(self)
            
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
                print(byType.pivot.reset_index().loc[dataidx].dropna())
                i+=1
            plt.show()
            return True
        
        byType.figscatter.canvas.mpl_connect('pick_event', onPick)
        
        byType.drawBar(self)
        
#         plt.show()
        
    
    def drawBar(self,ClbyTp=None):
        
        # by default, generate graph from instance variable Cluster-by-Type (DataFrame)
        if ClbyTp is None:
            ClbyTp=byType.ClbyTp
        # stacked bar plot
        ## xtab is a list of sorted clusters based on the number of places contained
        xtab=byType.clusters.groupby('Cluster').count().sort_values(by=byType.colName,ascending=False).index
        ## stacks is a list of selected features(Types of Event) used in K-means clustering
        stacks=byType.switch.set_index('boo').loc[True,'types'].values
        
        byType.axb.cla()
        xbar=np.arange(len(xtab))
        bottom=0
        bars=[]
        for feat in stacks:
            c=np.random.rand(4)
            p=byType.axb.bar(xbar,ClbyTp.loc[xtab,feat],color=c,bottom=bottom)
            bottom+=ClbyTp.loc[xtab,feat]
            bars.append(p)
        
        byType.axb.legend(bars,stacks)
        
        # Button: regenerate graph
        axperc = byType.figbar.add_axes([0.85, 0.03, 0.1, 0.06])
        bperc=Button(axperc,'to %')
        bperc.on_clicked(byType.toPercent)
        
        plt.show()
    
    def ClosenSave(self):
        plt.close('all')
        result=byType.clusters.sort_values(by='Cluster',ascending=False)
        result.to_csv("csv/"+byType.colName+"Cluster.csv",index=False,encoding='utf-8')
        
    def toPercent(self):
        def percent(arr):
            arr=[n/arr.sum() for n in arr]
            return arr
        pClbyTp=byType.ClbyTp.apply(percent,axis=1)
        byType.drawBar(self, pClbyTp)



g = byType()
g.init('Place')
g.drawScatter()
g.drawBar()


