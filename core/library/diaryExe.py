import pandas as pd
from library.diaryLibrary import ppl,ppl_plc, preprocess
import os
import glob

os.chdir("/Users/yumeng.zou/Google Drive/Freshyear/Summer/Research/Diary")
path="Spreadsheets/*.csv"
filenames=glob.glob(path)
 
diary=pd.DataFrame()
for f in filenames:
    yearlyDiary=pd.read_csv(f)
    diary=diary.append(yearlyDiary)
 
diary=preprocess(diary)
 
ppl(diary)
ppl_plc(diary)