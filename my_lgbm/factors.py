import pandas as pd
import numpy as np
from my.data import quote
from datetime import datetime
from datetime import time as dtime


def stamp2dt(timestamp):
    if len(str(timestamp)) > 10:
        timestamp = timestamp / pow(10, (len(str(timestamp)) - 10))

    dt = datetime.fromtimestamp(timestamp)
    return dt


def get_factors(bond_code, date):
    # calculate factors here
    data = quote.data(date, 265, "all", 0, 3, flatten=True)  # only quote data is used
    data = pd.DataFrame(data)
    data = data[data["symbol"] == bond_code.encode("gb2312")]
    # data = data.dropna()
    data.index = [stamp2dt(i) for i in data["local_time"]]
    data = data.resample("60s").last()
    data = data.loc[(data.index.time >= dtime(9, 30, 00)) & (data.index.time < dtime(16, 30, 00))]
    ap1 = np.array(data["ap1"])
    bp1 = np.array(data["bp1"])

    midprice = (ap1 + bp1) / 2
    midprice[np.isnan(midprice)] = 0

    av1 = np.array(data["av1"])
    av2 = np.array(data["av2"])
    av3 = np.array(data["av3"])
    av4 = np.array(data["av4"])
    av5 = np.array(data["av5"])

    bv1 = np.array(data["bv1"])
    bv2 = np.array(data["bv2"])
    bv3 = np.array(data["bv3"])
    bv4 = np.array(data["bv4"])
    bv5 = np.array(data["bv5"])

    imbalance1 = av1 / (av1 + bv1) - 0.5
    imbalance2 = av2 / (av2 + bv2) - 0.5
    imbalance3 = av3 / (av3 + bv3) - 0.5
    imbalance4 = av4 / (av4 + bv4) - 0.5
    imbalance5 = av5 / (av5 + bv5) - 0.5

    imbalance1[np.isnan(imbalance1)] = 0
    imbalance2[np.isnan(imbalance2)] = 0
    imbalance3[np.isnan(imbalance3)] = 0
    imbalance4[np.isnan(imbalance4)] = 0
    imbalance5[np.isnan(imbalance5)] = 0

    total_ask_size = av1 + av2 + av3 + av4 + av5
    total_bid_size = bv1 + bv2 + bv3 + bv4 + bv5
    avg_top_size = (av1 + av2) / 2
    size_diff = (total_ask_size - total_bid_size) / avg_top_size
    size_diff[np.isnan(size_diff)] = 0
    size_diff[np.isinf(size_diff)] = 0

    vwap = (ap1 * bv1 + bp1 * av1) / (av1 + bv1)
    price_diff = (vwap - midprice) / (av1 + bv1)
    price_diff[np.isnan(price_diff)] = 0

    wap_lag = np.diff(vwap)
    wap_lag = np.append(wap_lag, 0)
    wap_lag[np.isnan(wap_lag)] = 0

    size_diff_lag = np.diff(size_diff)
    size_diff_lag = np.append(size_diff_lag, 0)
    size_diff_lag[np.isnan(size_diff_lag)] = 0

    df = pd.DataFrame()
    df["imblance1"] = imbalance1
    df["imbalance2"] = imbalance2
    df["imbalance3"] = imbalance3
    df["imbalance4"] = imbalance4
    df["imbalance5"] = imbalance5
    df["size_diff"] = size_diff
    df["price_diff"] = price_diff
    df["wap_lag"] = wap_lag
    df["size_diff_lag"] = size_diff_lag
    df["midprice"] = midprice
    df.index = data.index

    return df
