'''
Created on Jul 25, 2017

@author: yumeng.zou
'''
import pandas as pd
import os
from library.diaryLibrary import StdType

os.chdir('/Users/yumeng.zou/Google Drive/Freshyear/Summer/Research/')

JQ25=pd.read_csv('JQ25 diary orig.csv')
StdTyped=StdType(JQ25)
    
StdTyped.to_csv('StdTyped.csv',index=False,encoding='utf-8')