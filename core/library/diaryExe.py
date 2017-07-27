import pandas as pd
from library.diaryLibrary import ppl,ppl_plc, preprocess
import os
import glob

# set directory
os.chdir("/Users/yumeng.zou/Google Drive/Freshyear/Summer/Research/Diary")

# concatenate all year diaries into one big dataframe
path="src/*.csv"
filenames=glob.glob(path)
diary=pd.DataFrame()
for f in filenames:
    yearlyDiary=pd.read_csv(f)
    diary=diary.append(yearlyDiary)

# clean up people's names, types, place names, and add event id
diary=preprocess(diary)

# make graphml file for people's network and people-place network
ppl(diary)
ppl_plc(diary)

# for further inspection
diary['Participants'].value_counts().sort_index().to_csv('csv/AllPeople.csv',encoding='utf-8')
diary['Place'].value_counts().sort_index().to_csv('csv/AllPlaces.csv',encoding='utf-8')
diary['Type'].value_counts().sort_index().to_csv('csv/AllTypes.csv',encoding='utf-8')