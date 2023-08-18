import numpy as np
import sklearn 
import pandas as pd
from sklearn.model_selection import KFold
import datetime


def calc_wap1(df):
    wap = (df['bid_price1']*df['ask_size1'] + df['ask_price1']*df['bid_size1'])/(df['bid_size1']+df['ask_size1'])
    return wap 

def calc_wap2(df):
    wap = (df['bid_price2']*df['ask_size2'] + df['ask_price2']*df['bid_size2'])/(df['bid_size2']+df['ask_size2'])
    return wap 

def cut_inf(data):
    data_new = 






    
@dataclass
class Data_obj:
    train_time: datetime
    test_time: datetime 
    bar: str
    labeling: 
    model: str
    





