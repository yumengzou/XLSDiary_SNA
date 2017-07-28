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
plt.rcParams['backend']='GTK3Agg'
plt.rcParams['font.family']='Microsoft YaHei'

diary=pd.read_csv("csv/Diary.csv")

class byType():
    
    def init(self,colName):
        
        byType.colName=colName
        
        EventCus=diary[['Event',colName]].dropna()
        EventType=diary[['Event','Type']].dropna()
        CusType=pd.merge(EventCus,EventType,how='left').drop("Event",axis=1)  
        CusType['Freq']=1
        CusType=CusType.groupby([colName,'Type']).sum().reset_index()
        gt4=CusType[colName].value_counts()>4 # boolean Series
        freqCus=CusType[colName].value_counts()[gt4].index
        byType.CusType=CusType.set_index(colName).ix[freqCus].reset_index()
        
        gt4=byType.CusType['Type'].value_counts()>4 # boolean Series
        freqType=byType.CusType['Type'].value_counts()[gt4].index
        byType.switch=pd.DataFrame({'types':freqType,'boo':True})
        
        filtered=byType.CusType.query("Type!='Go'&Type!='Visit'")
        byType.pivot=filtered.pivot(index=colName,columns='Type',values='Freq')
        
        byType.figscatter=plt.figure(figsize=(9,6))
        byType.axs = byType.figscatter.add_subplot(111, projection='3d')
        byType.axdraw = byType.figscatter.add_axes([0.85, 0.03, 0.1, 0.06])
        byType.axclose = byType.figscatter.add_axes([0.85, 0.9, 0.1, 0.06])
        byType.axcheck = byType.figscatter.add_axes([0.02, 0.3, 0.15, 0.5])
        
        byType.figbar=plt.figure(figsize=(7.2,4.8))
        byType.axb=byType.figbar.add_subplot(111)
        byType.axperc = byType.figbar.add_axes([0.85, 0.03, 0.1, 0.06])
          
    def filter(self):
        switch=byType.switch.set_index('boo')['types']
        thrownFeatures=['Type!='+repr(Type) for Type in switch[False]]
        
        if thrownFeatures:
            s='&'.join(thrownFeatures)
            filtered=byType.CusType.query(s)
        else:
            filtered=byType.CusType
        
        byType.pivot=filtered.pivot(index=byType.colName,columns='Type',values='Freq')
    
    # 3D scatter plot      
    def drawScatter(self):
        
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
        byType.axdraw.cla()
        bdraw=Button(byType.axdraw,'Draw')
        bdraw.on_clicked(byType.drawScatter)
        
        # Button: close graph and store csv
        byType.axclose.cla()
        bclose=Button(byType.axclose,'Close')
        bclose.on_clicked(byType.ClosenSave)
        
        # Check Button: filter features
        byType.axcheck.cla()
        check = CheckButtons(byType.axcheck, byType.switch['types'], byType.switch['boo'])
        
        def checkFilter(label):
            switch=byType.switch.set_index('types')['boo']
            switch[label]=not switch[label]
            byType.switch['boo']=switch.values
            
            byType.filter(self)
            
        check.on_clicked(checkFilter)
        
        # mouse event: a pie chart for each point
        def pickPlace(event):
            
            if event.artist != points:
                return True
                
            N = len(event.ind)
            if not N:
                return True
            
            for i,dataidx in enumerate(event.ind):
                if i==3:
                    break
                # pie chart
                figpie,axpie=plt.subplots()
                ser=byType.pivot.reset_index().loc[dataidx].dropna()
                Labels=ser[1:].index
                size=[round(n*100/ser[1:].sum()) for n in ser[1:].values]
                Explode=[0.1 if p==max(size) else 0 for p in size]
                axpie.pie(size, explode=Explode, labels=Labels, autopct='%1.1f%%', shadow=True, startangle=90)
                axpie.set_title(ser[0])
            plt.show()
            return True
        
        byType.figscatter.canvas.mpl_connect('pick_event', pickPlace)
        
        byType.drawBar(self)
          
    # stacked bar plot
    def drawBar(self,ClbyTp=None):
        
        # by default, generate graph from instance variable Cluster-by-Type (DataFrame)
        if ClbyTp is None:
            ClbyTp=byType.ClbyTp
        
        ## xtab is a list of sorted clusters based on the number of places contained
        xtab=byType.clusters.groupby('Cluster').count().sort_values(by=byType.colName,ascending=False).index
        ## stacks is a list of selected features(Types of Event) used in K-means clustering
        stacks=byType.switch.set_index('boo').loc[True,'types'].values
        
        byType.axb.cla()
        xbar=np.arange(len(xtab))
        bottom=0
        bars=[]
        for feat in stacks:
            p=byType.axb.bar(xbar,ClbyTp.loc[xtab,feat],color=np.random.rand(4),bottom=bottom,picker=2,align='center')
            bottom+=ClbyTp.loc[xtab,feat]
            bars.append(p)
        byType.axb.set_xticklabels(xtab)
        byType.axb.legend(bars,stacks)
        
        # Button: regenerate graph
        byType.axperc.cla()
        bperc=Button(byType.axperc,'to %')
        bperc.on_clicked(byType.toPercent)
        
        # mouse event: all points contained by the clicked cluster(bar)
        texts=[]
        for i,x in enumerate(xbar):
            s=byType.clusters[byType.clusters['Cluster']==xtab[i]]['Place']
            t=byType.axb.text(x-0.4,0.9,s,visible=False)
            texts.append(t)
        
        def pickCluster(event):
                    
            if event.artist not in byType.axb.patches:
                return True
                        
            xr,yr=event.artist.xy
            texts[int(np.ceil(xr))].set_visible(not texts[int(np.ceil(xr))].get_visible())
            
            plt.show()
            
            return True
        
        byType.figbar.canvas.mpl_connect('pick_event', pickCluster)
        
        plt.show()
    
    def ClosenSave(self):
        byType.figscatter.clear()
        byType.figbar.clear()
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


