'''
Created on Jul 30, 2017

@author: yumeng.zou
'''
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import sklearn.decomposition as skl
import pyecharts as pch
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.widgets import Button, CheckButtons

os.chdir("/Users/yumeng.zou/Google Drive/Freshyear/Summer/Research/Diary/")
# plt.rcParams['backend']='GTK3Agg'
# plt.rcParams['font.family']='Microsoft YaHei'

diary=pd.read_csv("csv/Diary.csv")

def percent(arr):
    arr=[n/arr.sum() for n in arr]
    return arr

class Cluster():
    
    def init(self,sampleCol,featureCol):
        
        Cluster.sampleCol=sampleCol
        Cluster.featureCol=featureCol
        
        EventSamp=diary[['Event',sampleCol]].dropna()
        EventFeat=diary[['Event',featureCol]].dropna()
        SampFeat=pd.merge(EventSamp,EventFeat,how='left').drop("Event",axis=1)
        SampFeat['Freq']=1
        SampFeat=SampFeat.groupby([sampleCol,featureCol]).sum().reset_index()
        
        # Sample-Feature-Freq table: select the top frequent instances in the user-selected column
        gt4=SampFeat[sampleCol].value_counts()>1 # boolean Series
        freqSamp=SampFeat[sampleCol].value_counts()[gt4].index
        Cluster.SampFeat=SampFeat.set_index(sampleCol).ix[freqSamp].reset_index()
        
        
        # Sample-Feature-Freq pivot table
        Cluster.pivot=Cluster.SampFeat.pivot(index=sampleCol,columns=featureCol,values='Freq').apply(percent,axis=1)
        
        # features-boolean Series: select the top frequent types as filter-able
#         gt4=Cluster.SampFeat[featureCol].value_counts()>5 # boolean Series
        freqFeat=Cluster.SampFeat[featureCol].value_counts()[:10].index
        Cluster.switch=pd.DataFrame({featureCol:freqFeat,'boo':True})
        
        # 3D scatter plot
#         Cluster.figscatter=plt.figure(figsize=(9,6))
#         Cluster.axs = Cluster.figscatter.add_subplot(111, projection='3d')
#         Cluster.axdraw = Cluster.figscatter.add_axes([0.85, 0.03, 0.1, 0.06])
#         Cluster.axclose = Cluster.figscatter.add_axes([0.85, 0.9, 0.1, 0.06])
#         Cluster.axcheck = Cluster.figscatter.add_axes([0.02, 0.3, 0.15, 0.6])
        
        # bar plot
#         Cluster.figbar=plt.figure(figsize=(7.2,4.8))
#         Cluster.axb=Cluster.figbar.add_subplot(111)
#         Cluster.axperc = Cluster.figbar.add_axes([0.85, 0.03, 0.1, 0.06])
    
    # regenerate the Sample-Feature-Freq pivot table after switch is altered
#     def filter(self):
#         
#         switch=Cluster.switch[Cluster.switch['boo']==False][Cluster.featureCol]
#         
#         if isinstance(switch, pd.Series) and switch.empty:
#             filtered=Cluster.SampFeat
#         else:
#             thrownFeatures=[Cluster.featureCol+'!='+repr(feat) for feat in switch]
#             s='&'.join(thrownFeatures)
#             filtered=Cluster.SampFeat.query(s)
#         
#         Cluster.pivot=filtered.pivot(index=Cluster.sampleCol,columns=Cluster.featureCol,values='Freq').replace(np.nan,0)
    
    # draw 3D scatter plot      
    def draw(self):
        
#         Cluster.filter(self)
        
        mat=Cluster.pivot.replace(np.nan,0).as_matrix()
        cluster=KMeans(n_clusters=6).fit(mat)
        
        # clustering result
        Cluster.clusters=pd.DataFrame({Cluster.sampleCol:Cluster.pivot.index,'Cluster':cluster.labels_})
        
        # Cluster-Sample-Feature table: explore the meaning/constitution of a cluster
        ClSpFt=pd.merge(Cluster.SampFeat,Cluster.clusters,how='left').groupby('Cluster')
        ForEachCluster=lambda x : x.pivot(index=Cluster.sampleCol,columns=Cluster.featureCol,values='Freq').apply(percent).replace(np.nan,0)
        Cluster.ClSpFt=ClSpFt.apply(ForEachCluster)
        print(Cluster.ClSpFt)
        
        
        pca=skl.PCA(n_components=3).fit(mat)
        xyz=pca.transform(mat)
        xl,yl,zl=pca.explained_variance_ratio_
        scatter=pch.Scatter3D('Clustering ' + Cluster.sampleCol +' by ' + Cluster.featureCol, is_grid=True)
        
        for i, group in pd.concat([Cluster.clusters,pd.DataFrame(xyz)],axis=1).groupby('Cluster'):
            scatter.add('Cluster '+str(i),xyz[group.index].tolist(),xaxis_name=xl, yaxis_name=yl,grid_top="60%",legend_pos='80%')
        
        
        radar=pch.Radar('Radar Chart of ' + Cluster.sampleCol + ' on ' + Cluster.featureCol,is_grid=True)
        schema=[(feat,1) for feat in Cluster.switch[Cluster.featureCol]]
        radar.config(schema)
        
        for i in np.arange(6):
            data=Cluster.ClSpFt.loc[i].as_matrix().tolist()
            c=['#50a3ba', '#eac763', '#d94e5d','#4e79a7','#f9713c','#b3e4a1']
            radar.add('Cluster '+str(i), data, item_color=c[i],legend_pos='80%')
#             print(i,cluster.pivot(index=Cluster.sampleCol,columns=Cluster.featureCol,values='Freq'))
#         scatter.grid(radar.get_series(),grid_bottom="60%")
        
        radar.render('pyecharts_radar.html')
        
        scatter.render('pyecharts_scatter.html')
#         x,y,z=*pca.transform(mat)
        
#         scatter=Scatter3D('Clustering ' + Cluster.sampleCol +' based on the ' + Cluster.featureCol, width=1200, height=600)
#         scatter.add('love',pca.transform(mat).tolist())
#         scatter.render('pyecharts_scatter.html')
        
#         Cluster.axs.cla()
#         points=Cluster.axs.scatter(y,x,z,c=cluster.labels_,s=50,alpha=0.5,picker=3)
#         Cluster.axs.set_title('Clustering ' + Cluster.sampleCol +' based on the ' + Cluster.featureCol)
#         
#         # axes labels
#         xl,yl,zl=pca.explained_variance_ratio_
#         Cluster.axs.set_xlabel('{:.2f}%'.format(xl*100))
#         Cluster.axs.set_ylabel('{:.2f}%'.format(yl*100))
#         Cluster.axs.set_zlabel('{:.2f}%'.format(zl*100))
#         
#         # Button: regenerate graph
#         Cluster.axdraw.cla()
#         bdraw=Button(Cluster.axdraw,'Draw')
#         bdraw.on_clicked(Cluster.draw)
#         
#         # Button: close graph and store csv
#         Cluster.axclose.cla()
#         bclose=Button(Cluster.axclose,'Close')
#         bclose.on_clicked(Cluster.ClosenSave)
#         
#         # Check Button: filter features
#         Cluster.axcheck.cla()
#         check = CheckButtons(Cluster.axcheck, Cluster.switch[Cluster.featureCol], Cluster.switch['boo'])
#         
#         def checkFilter(label):
#             
#             boo=Cluster.switch.set_index(Cluster.featureCol)['boo']
#             boo[label]=not boo[label]
#             Cluster.switch['boo']=boo.values
#             
#         check.on_clicked(checkFilter)
#         
#         # mouse event: a pie chart for each point on click
#         def pickCus(event):
#             
#             if event.artist != points:
#                 return True
#                 
#             N = len(event.ind)
#             if not N:
#                 return True
#             
#             for i,dataidx in enumerate(event.ind):
#                 if i==3:
#                     break
#                 # pie chart
#                 figpie,axpie=plt.subplots()
#                 ser=Cluster.pivot.reset_index().loc[dataidx].dropna()
#                 Labels=ser[1:].index
#                 size=[round(n*100/ser[1:].sum()) for n in ser[1:].values]
#                 Explode=[0.1 if p==max(size) else 0 for p in size]
#                 axpie.pie(size, explode=Explode, labels=Labels, autopct='%1.1f%%', shadow=True, startangle=90)
#                 axpie.set_title(ser[0])
#             
#             plt.show()
#             return True
#         
#         Cluster.figscatter.canvas.mpl_connect('pick_event', pickCus)
        
#         Cluster.drawBar(self)
          
    # draw stacked bar plot
    def drawBar(self,ClbyFt=None):
        
#         Cluster.axb.cla()
        
        # by default, generate graph from instance variable Cluster-Feature-Count (DataFrame)
        if ClbyFt is None:
            ClbyFt=Cluster.ClbyFt
        
        ## xtab is a list of sorted clusters based on the number of places contained
        xtab=Cluster.clusters.groupby('Cluster').count().sort_values(by=Cluster.sampleCol,ascending=False).index
        ## stacks is a list of selected features used in K-means clustering
        stacks=Cluster.switch.set_index('boo').loc[True,Cluster.featureCol].values
        
#         xbar=np.arange(len(xtab))
#         bottom=0
#         bars=[]
#         for feat in stacks:
#             p=Cluster.axb.bar(xbar,ClbyFt.loc[xtab,feat],color=np.random.rand(4),bottom=bottom,picker=5,align='center')
#             bottom+=ClbyFt.loc[xtab,feat]
#             bars.append(p)
#         Cluster.axb.set_xticklabels(xtab)
#         Cluster.axb.legend(bars,stacks)
#         
#         # Button: regenerate graph
#         Cluster.axperc.cla()
#         bperc=Button(Cluster.axperc,'to %')
#         bperc.on_clicked(Cluster.toPercent)
        
        # mouse event: display all instances in a cluster on click
#         texts=[]
#         for i,x in enumerate(xbar):
#             s='\n'.join(Cluster.clusters[Cluster.clusters['Cluster']==xtab[i]][Cluster.sampleCol].values)
#             t=Cluster.axb.text(x-0.4,20,s,visible=False)
#             texts.append(t)
        
#         def pickCluster(event):
#                     
#             if event.artist not in Cluster.axb.patches:
#                 return True
#             
#             xr,yr=event.artist.xy
#             texts[int(np.ceil(xr))].set_y(yr)
#             texts[int(np.ceil(xr))].set_bbox({'facecolor':'white','alpha':0.5})
#             texts[int(np.ceil(xr))].set_visible(not texts[int(np.ceil(xr))].get_visible())
#             
#             plt.draw()
#             
#             return True
#         
#         Cluster.figbar.canvas.mpl_connect('pick_event', pickCluster)
#         
#         plt.show()
    
    # normalize the bar plot
    def toPercent(self):
        pClbyFt=Cluster.ClbyFt.apply(percent,axis=1)
        Cluster.drawBar(self, pClbyFt)
    
    # clear and close all plt figures and save the col-Type-Freq table and the clustering result to csv
    def ClosenSave(self):
#         Cluster.figscatter.clear()
#         Cluster.figbar.clear()
#         plt.close('all')
        result=Cluster.clusters.sort_values(by='Cluster',ascending=False)
        result.to_csv("csv/"+Cluster.sampleCol+"Cluster_on"+Cluster.featureCol+".csv",index=False,encoding='utf-8')
        Cluster.SampFeat.to_csv('csv/'+Cluster.sampleCol+Cluster.featureCol+'Freq.csv',index=False,encoding='utf-8')



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
