
# coding: utf-8

import pandas as pd
import mafan as mf
import re
import networkx as nx


"""
--- preprocess: delete empty rows/columns, standardize names, and make event id ---
@diary = hand-typed spreadsheet
export the preprocessed diary into a csv
return preprocessed diary
"""
def preprocess(diary):
    # clean the data
    diary.dropna(axis=[0,1],how='all',inplace=True)
    diary['Participants']=StdNm(diary['Participants'])
    diary=StdType(diary)
    
    # make event index from Type
    event_iter=0
    event_idx=[]
    for boo in diary['Type'].notnull():
        if boo:
            event_iter+=1
        event_idx.append(event_iter)
    
    diary["Event"]=event_idx
    
    # export into a csv
    diary.to_csv("Diary.csv",index=False,encoding='utf-8')
    
    return diary


"""
--- unimodal network of who-knows-who ---
@diary = preprocessed DataFrame
export a graphml file of nodes and edges, a csv of all people appeared in the diary
return
"""
def ppl(diary):
    
    # make people-event DataFrame
    ppl_evt=diary[['Event','Participants']].dropna(subset=['Participants'])
    
    evtXie=pd.DataFrame({"Event":ppl_evt['Event'].unique(),"Participants":"謝蘭生"})
    ppl_evt=ppl_evt.append(evtXie,ignore_index=True)

    # make edgelist from people-event DF
    el=ppl_evt.merge(ppl_evt,on="Event")
    el=el.drop("Event",axis=1).rename(columns={"Participants_x":"Source","Participants_y":"Target"})
    el=el.query("Source<Target")
    el["Weight"]=1

    # Calculate Weight of edges with Groupby
    edgelist=el.groupby(["Source","Target"]).sum().reset_index()
    
    # export edges into a graphml file
    G = nx.from_pandas_dataframe(edgelist, source="Source", target="Target", edge_attr="Weight")
    nx.write_graphml(G, "Graph/ppl.graphml", encoding="utf-8")
    
    # make a series of unique names as the index and columns of the adjacency matrix
    ppl_ls=list(diary['Participants'].unique())
    ppl_ls.append("謝蘭生")
    pplNodes=pd.Series(ppl_ls)

    pplNodes.to_csv('AllPeople.csv',index=False,encoding='utf-8')
    
    return



"""
--- people-place bimodal network ---
@diary = preprocessed DataFrame
export a graphml file of nodes and edges
return
"""
def ppl_plc(diary):
    
    evt_plc=diary[['Event','Place']].dropna()
    
    evt_ppl=diary[['Event','Participants']]
    Xie=pd.DataFrame({'Event':diary['Event'],'Participants':'謝蘭生'})
    evt_ppl=evt_ppl.append(Xie,ignore_index=True)
    
    ppl_plc=pd.merge(evt_plc,evt_ppl,how='left').dropna().drop('Event',axis=1)
    ppl_plc['Weight']=1
    
    edges=ppl_plc.groupby(['Place','Participants']).sum().reset_index()
    edges.rename(columns={'Participants':'Source','Place':'Target'},inplace=True)
    
    G = nx.from_pandas_dataframe(edges, source="Source", target="Target", edge_attr="Weight")
    nx.write_graphml(G, "Graph/ppl_plc.graphml", encoding="utf-8")
#     edges.to_csv("NdEg/ppl_plc_Edges.csv",index=False,encoding='utf=8')
    
    return



"""
--- Standardize People's Names -- One to Many Transformation ---
@nonstd = Series of non-standard names
return standardized names in a Series
"""
def StdNm (nonstd=None):
    
    ## make a DataFrame indexed by standardized name and
    ## contains columns "FirstName", "OtherNames", "OtherNames1"
    std=pd.read_csv("StandardNames.csv")
    std["FullName"]=std["LastName"]+std["FirstName"]
    std["FullName"].fillna(method="ffill",inplace=True)
    std.set_index("FullName",inplace=True)

    #### Simplified characters for LastName happen sometimes
    std["ConvLast"]=[sTR if pd.isnull(sTR)
                     else mf.simplify(sTR) if mf.is_traditional(sTR)
                     else mf.tradify(sTR)
                     for sTR in std["LastName"]]
    std["LastOth"]=std["LastName"].fillna(method="ffill")+std['OtherNames']
    std["ConvFst"]=std["ConvLast"].fillna(method="ffill")+std['FirstName']
    std["ConvOth"]=std["ConvLast"].fillna(method="ffill")+std['OtherNames']
    std.drop(["Details","Studio","LastName","ConvLast"],axis=1,inplace=True)
    
    ## make a dataframe of {key: alternative names, value: standard names} with
    ## unique keys and overlapping values
    map_df=pd.DataFrame()
    
    for colName in list(std.columns):
        df=pd.DataFrame({"key":std[colName],"value":std.index})
        map_df=map_df.append(df,ignore_index=True)
        
    map_df.dropna(inplace=True)

    ## Standardize names in the given Series
    map_dict=map_df.set_index('key').to_dict()['value']
    def standardize_names(participant):
        if participant in map_dict:
            return map_dict[participant]
        else:
            return participant

    ans=nonstd.map(standardize_names)

    return ans



"""
--- Standardize Type and Direction columns in a given diary ---
@diary = pandas DataFrame
export standardized diary as a csv file
return standardized diary
"""
def StdType(diary):
    
    def pattern1(ser):
        
        try:
            ser['Direction']=ser['Direction'].lower()
        except:
            AttributeError
            pass
        
        if ser['Type']=='Visiting' or ser['Type']=='Visited':
            if ser['Type']=='Visiting':
                ser['Direction']='to'
            else:
                ser['Direction']='from'
            ser['Type']='Visit'
        
        if ser['Type']=='Writing_For' or ser['Type']=='Painting_For':
            ser['Type']='Art'
            ser['Direction']='for'
        
        if ser['Type']=='Inviting' or ser['Type']=='Invited':
            if ser['Type']=='Inviting':
                ser['Direction']='to'
            else:
                ser['Direction']='from'
            ser['Type']='Invitation'
        
        if ser['Type']=='Gift_received' or ser['Type']=='Gift_sent':
            if ser['Type']=='Gift_received':
                ser['Direction']='from'
            else:
                ser['Direction']='to'
            ser['Type']='Gift'
        
        occasions=['道喜','道壽','Celebrate','Send_off','拜年','道壽喜','吊']
        if ser['Type'] in occasions:
            ser=['Occasion',ser['Type']]
        
        return ser

    def pattern2(Type):
        
        try:
            Type=Type.title()
        except:
            AttributeError
            return Type
        
        if Type=='Painting' or Type=='Writing' or Type=='寫字' or Type=='art':
            Type='Art'
        
        if Type=='Go_places':
            Type='Go'
        
        if Type=='Theater' or Type=='彈琴':
            Type='Recreation'
        
        if Type=='Meal' or Type=='飲' or Type=='meal' or Type=='Drinking' or Type=='食荔子':
            Type='Banquet'
        
        if Type=='談禪':
            Type='Chat'
        
        if Type=='候':
            Type='Visit'
        
        if Type=='Ceremony':
            Type='Occasion'
        
        if Type=='Ldoge':
            Type='Lodge'
        
        if Type=='Gazetteer':
            Type='Administrative'
        
        if Type=='静坐' or Type=='習静' or Type=='坐禪' or Type=='齋':
            Type='Religious'
            
        return Type
    
    def pattern3(Note):
        
        try:
            Note=Note.title()
        except:
            AttributeError
            return Note
        
        if Note=='No':
            Note='Unsuccessful'
        if Note=='Yes':
            Note=None
        
        return Note
    
    diary[['Type','Direction']]=diary[['Type','Direction']].apply(pattern1,axis=1)
    diary['Type']=diary['Type'].map(pattern2)
    diary['Note']=diary['Note'].map(pattern3)
    
    return diary



"""
--- Extract weather information from a txt file ---
@txt_path   = path of the diary text file
@diary_path = path of the diary csv
export a csv file of English date, Chinese date, and extracted weather
return
"""
def extract_weather(txt_path,diary_path):

    # length is the maximum length of strings representing weather

    diary=pd.read_csv(diary_path)
    text=open(txt_path,'r',encoding='utf-8').read()
    weather=pd.DataFrame(re.findall("\n(.{2}日)(.{1,25}?)。{1}",text))
    # {m,n}? captures matched patterns from length m to length n, as few as possible
    # re.findall("regex()()") returns [($1,$2),($1,$2),...]
    weather.rename(columns={0:'日期',1:'weather'},inplace=True)

    # make a DataFrame of {Date, 日期, weather}
    DateVWeather=pd.DataFrame({'date':diary['Date'].dropna().unique()}).join(weather,how='outer')
    DateVWeather.to_csv('DateVWeather.csv',encoding='utf-8',index=False)

    print("Extracted " + str(len(weather)) +
          " weathers and stored as DateVWeather.csv in the current working directory.\n" + 
          "Length of the diary's date is " + str(len(diary['Date'].dropna().unique())) + 
          "Please correct all the errors and give them back to diaryLibrary.make_weather()")
    
    return



"""
--- Put weather information into the diary spreadsheet ---
@weather_path = path to the weather csv
@diary_path   = path to the diary csv
export a csv file of diary with weather
"""
def make_weather(weather_path,diary_path):

    weather=pd.read_csv(weather_path)
    diary=pd.read_csv(diary_path)
    
    if len(weather['weather'].dropna())!=len(diary['Date'].dropna().unique()):
        print("Check errors in the weather DataFrame.")
        return
        
    else:
        date=pd.DataFrame({'date':diary['Date'].dropna().drop_duplicates()}).reset_index()
        merged=pd.merge(date,weather,how='left')
        indexed_weather=merged.drop(['date','日期'],axis=1).set_index('index')
        diary['Weather']=indexed_weather
        diary.to_csv(diary_path,encoding='utf-8',index=False)

        return
    return
