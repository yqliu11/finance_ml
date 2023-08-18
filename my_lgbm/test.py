import numpy as np
import pandas as pd
import multiprocessing as mp
from my.data.basic_func import get_basic_data
from my.data.meta_api import get_before_trade_info
from datetime import time as dtime
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from optuna.integration import LightGBMPruningCallback
import optuna


def get_dates(start_dt, end_dt):
    table_name = "bond_calendar"
    df = get_basic_data(20230728, table_name)
    cal_list = df[df.exchmarket == "NIB"].trade_dt
    dates = []
    for i in cal_list:
        if (int(i) >= start_dt) & (int(i) <= end_dt):
            dates.append(int(i))
    dates_not_in = [
        20220129,
        20220130,
        20221107,
        20221202,
        20221216,
        20221219,
        20221231,
        20230120,
        20230128,
        20230129,
        20230215,
        20230216,
        20230410,
        20230423,
        20230506,
        20230517,
        20230625,
    ]
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

    df = factors.get_factors("220220.IB", 20230418)  # to get the shape of factors
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


def split(Fx, Fy, period, dates):
    kk = round(len(dates) / period)
    a, b, c = np.shape(Fx)
    train_xlist = np.zeros((kk, period * 300, c))
    train_ylist = np.zeros((kk, period * 300))
    test_xlist = np.zeros((kk, 300, c))
    test_ylist = np.zeros((kk, 300))

    for i in range(kk):
        train = list(range(i, period + i))
        train_x = np.vstack(Fx[train, :])
        train_y = np.hstack(Fy[train, :])
        test_x = Fx[period + i]
        test_y = Fy[period + i]
        train_xlist[i, :, :] = train_x
        train_ylist[i, :] = train_y
        test_xlist[i, :, :] = test_x
        test_ylist[i, :] = test_y
    return train_xlist, train_ylist, test_xlist, test_ylist


Fx, Fy = read_data(20220101, 20221231)
start_dt = 20220101
end_dt = 20221231
dates = get_dates(start_dt, end_dt)
# train_xlist, train_ylist, test_xlist, test_ylist = split(Fx, Fy, 200, dates)

period = 100

kk = len(dates) - period - 1
a, b, c = np.shape(Fx)
train_xlist = np.zeros((kk, period * 300, c))
train_ylist = np.zeros((kk, period * 300))
test_xlist = np.zeros((kk, 300, c))
test_ylist = np.zeros((kk, 300))

for i in range(kk):
    train = list(range(i, period + i))
    train_x = np.vstack(Fx[train, :])
    train_y = np.hstack(Fy[train, :])
    test_x = Fx[period + i]
    test_y = Fy[period + i]
    # delete nan or infs
    train_y[(train_y > 100) | (train_y < -100)] = 0
    train_y[np.isnan(train_y)] = 0

    test_y[(test_y > 100) | (test_y < -100)] = 0
    test_y[np.isnan(test_y)] = 0

    train_xlist[i, :, :] = train_x
    train_ylist[i, :] = train_y
    test_xlist[i, :, :] = test_x
    test_ylist[i, :] = test_y

i = 1
X_train = train_xlist[i]
X_test = test_xlist[i]
y_train = train_ylist[i]
y_test = test_ylist[i]
# y_train = y_train.ravel()
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


def lgb_optuna(trial, train_x, train_y, test_x, test_y):
    param = {
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "max_bin": 100,
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 10000),
        "learning_rate": 0.01,
        "bagging_fraction": 0.95,
        "bagging_freq": 5,
        "bagging_seed": 66,
        "feature_fraction": trial.suggest_loguniform("feature_fraction", 0.55, 0.99),
        "feature_fraction_seed": 66,
        # loss
        "lambda_l1": trial.suggest_discrete_uniform("lambda_l1", 0.0, 10.0, 0.1),
        "lambda_l2": trial.suggest_discrete_uniform("lambda_l2", 0.0, 10.0, 0.1),
        "min_gain_to_split": rest_dict["min_gain_to_split"],
        # greedy
        "min_sum_hessian_in_leaf": trial.suggest_discrete_uniform("min_sum_hessian_in_leaf", 0.55, 20.0, 0.1),
        # object-metric
        "objective": "regression",
        "metric": "rmse",
        "n_jobs": 25,
        "boosting": "gbdt",
        "verbose": 1,
        "early_stopping_rounds": 50,
        "n_estimators": 500,
    }
    model = lgb.LGBMRegressor(**param)
    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], early_stopping_rounds=50, verbose=200)
    pred_ = model.predict(test_x)
    loss = np.sqrt(mean_squared_error(test_x, np.round(np.expm1(pred_))))
    return loss
