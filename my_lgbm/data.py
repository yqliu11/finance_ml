import numpy as np
import pandas as pd
import multiprocessing as mp
from my.data.basic_func import get_basic_data
from my.data.meta_api import get_before_trade_info
from datetime import time as dtime


def get_dates(start_dt, end_dt):
    table_name = "bond_calendar"
    df = get_basic_data(20230728, table_name)
    cal_list = df[df.exchmarket == "NIB"].trade_dt
    dates = []
    for i in cal_list:
        if (int(i) >= start_dt) & (int(i) <= end_dt):
            dates.append(int(i))
    dates_not_in = [20220129, 20220130, 20221107, 20221202, 20221216,
        20221219, 20221231, 20230120, 20230128, 20230129, 20230215, 20230216,  20230410, 20230423, 20230506, 20230517, 20230625]
    for j in dates_not_in:
        if j in dates:
            dates.remove(j)
    return dates


def resample(df, period):
    # period = 60s
    df = df.resample(period).last()
    return df


def label(series, method):
    if method == "logdiff":
        return np.log(series).diff() * 1e4
    if method == "diff":
        return np.diff(series)


def clean_ourliers(df):
    df[df["ylabel"] > 10] = 0  # clean outliers
    return df


def read_data(start_dt, end_dt):
    dates = get_dates(start_dt, end_dt)
    import factors
    df = factors.get_factors('220220.IB', 20230418)  # to get the shape of factors
    a, b = np.shape(df)
    Fx = np.zeros((len(dates), 300, b))  # 300 mins every day a= 300
    Fy = np.zeros((len(dates), 300))
    for i in range(len(dates)):
        date = dates[i]
        # print(date)
        df_mc = get_before_trade_info(date, "bond_main_contract")
        mc_dict = dict(zip(df_mc.label, df_mc.securityid))
        if "CDB10Y01.IB" in mc_dict:
            bond_code = mc_dict["CDB10Y01.IB"]
        else:
            bond_code = "220220.IB"
        df = factors.get_factors(bond_code, date)
        # df = df.resample("60s").last()
        df["ylabel"] = label(df["midprice"], "logdiff")
        # delete data between 11:30 and 13:30
        df = df.drop(df[(df.index.time >= dtime(11, 30, 00)) & (df.index.time < dtime(13, 30, 00))].index)
        df["mins"] = [i for i in range(len(df))]
        Fy[i, :] = df.ylabel
        del df["ylabel"]
        del df["midprice"]
        Fx[i, :] = df
    return Fx, Fy


def split(Fx, Fy, period):
    kk = round(len(dates)/period)
    a, b, c= np.shape(Fx)
    train_xlist = np.zeros((kk, period*300, c))
    train_ylist = np.zeros((kk, period*300))
    test_xlist = np.zeros((kk, 300, c))
    test_ylist = np.zeros((kk, 300))

    for i in range(kk):
        train = list(range(i, period+i))
        train_x = np.vstack(Fx[train,:])
        train_y = np.hstack(Fy[train,:])
        test_x = Fx[period+i]
        test_y = Fy[period+i]
        train_xlist[i, :, :] = train_x
        train_ylist[i, : ] = train_y
        test_xlist[i,:,:] = test_x
        test_ylist[i,:] = test_y
    return train_xlist, train_ylist, test_xlist, test_ylist



# 调用LightGBM模型，使用训练集数据进行训练（拟合）
# Add verbosity=2 to print messages while running boosting
train_X, test_X, train_y, test_y = train_test_split(Fx_resam, Fy_resam, test_size=0.1)
lgb_train = lgb.Dataset(train_X, train_y)
lgb_test  = lgb.Dataset(test_X, test_y)
my_model = lgb.LGBMRegressor(objective="regression", num_leaves=10, learning_rate=0.1, n_estimators=20, verbosity=2)
my_model.fit(np.array(train_X[0:5]).transpose(), train_y[0:5], verbose=False)
# my_model.fit(train_X[1].transpose(), train_y[1], verbose=False)






total_X = 
# conenct the arrays
for i in range(len(dates)):


