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

os.chdir("/Users/yumeng.zou/Google Drive/Freshyear/Summer/Research/Diary/")

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
        
    
    # draw 3D scatter plot      
    def draw(self):
        
        mat=Cluster.pivot.replace(np.nan,0).as_matrix()
        cluster=KMeans(n_clusters=6).fit(mat)
        
        # clustering result
        Cluster.clusters=pd.DataFrame({Cluster.sampleCol:Cluster.pivot.index,'Cluster':cluster.labels_})
        
        # Cluster-Sample-Feature table: explore the meaning/constitution of a cluster
        ClSpFt=pd.merge(Cluster.SampFeat,Cluster.clusters,how='left').groupby('Cluster')
        ForEachCluster=lambda x : x.pivot(index=Cluster.sampleCol,columns=Cluster.featureCol,values='Freq').apply(percent).replace(np.nan,0)
        Cluster.ClSpFt=ClSpFt.apply(ForEachCluster)        
        
        pca=skl.PCA(n_components=3).fit(mat)
        xyz=pca.transform(mat)
#         xl,yl,zl=pca.explained_variance_ratio_
        
        # 3D scatter plot
        scatter=pch.Scatter3D('Clustering ' + Cluster.sampleCol +' by ' + Cluster.featureCol, is_grid=True,
                              width=1200,height=600)
        
        for i, group in pd.concat([Cluster.clusters,pd.DataFrame(xyz)],axis=1).groupby('Cluster'):
            scatter.add('Cluster '+str(i),xyz[group.index].tolist(),
                        grid_right="75%",legend_top="10%",legend_pos="0%")
        
        # Radar chart
        radar=pch.Radar('Radar Chart of ' + Cluster.sampleCol + ' on ' + Cluster.featureCol, title_pos="60%")
        schema=[(feat,1) for feat in Cluster.switch[Cluster.featureCol]]
        radar.config(schema)
        
        for i in np.arange(6):
            data=Cluster.ClSpFt.loc[i].as_matrix().tolist()
            c=['#50a3ba', '#eac763', '#d94e5d','#4e79a7','#f9713c','#b3e4a1']
            radar.add('Cluster '+str(i), data, item_color=c[i],legend_pos='50%',legend_top="10%",
                      is_area_show=True,area_color=c[i],area_opacity=0.5)

        radar.render('Graph/radar_'+Cluster.sampleCol+Cluster.featureCol+'.html')
         
        scatter.render('Graph/scatter'+Cluster.sampleCol+Cluster.featureCol+'.html')
#         scatter.grid(radar.get_series(),grid_left="75%")
        
#         scatter.render('TEST.html')
        
        Cluster.ClSpFt.to_csv('csv/Cluster'+Cluster.sampleCol+'_on'+Cluster.featureCol+'.csv',encoding='utf-8')


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
