import pandas as pd
import numpy as np
from my.data.basic_func import get_basic_data
from my.data.meta_api import get_before_trade_info
from finance_ml.labeling import get_barrier_labels
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import factors

start_dt = 20220103
end_dt = 20230429

table_name = "bond_calendar"
df = get_basic_data(20230728, table_name)
cal_list = df[df.exchmarket == "NIB"].trade_dt
dates = []
for i in cal_list:
    if (int(i) >= start_dt) & (int(i) <= end_dt):
        dates.append(int(i))
dates_not_in = [20221219, 20221231, 20230120, 20230129, 20230423, 20230506, 20230625]
for i in dates_not_in:
    dates.remove(i)

Fx = []
Fy = []
for date in dates:
    df_mc = get_before_trade_info(date, "bond_main_contract")
    mc_dict = dict(zip(df_mc.label, df_mc.securityid))
    if "CDB10Y01.IB" in mc_dict:
        bond_code = mc_dict["CDB10Y01.IB"]
    else:
        bond_code = "220220.IB"
    terms, dfy = factors.get_factors(bond_code, date)
    labels = get_barrier_labels(
        dfy,
        trgt=5e-6,
        sltp=3,  # width
        seconds=100,  # horizontal time range seconds
        sign_label=False,  # if  True, label based on sign of return when touching vertical line
        num_threads=mp.cpu_count(),
    )
    print(date)
    Fx.append(terms)
    Fy.append(labels)
Fx = np.array(Fx)
Fy = np.array(Fy)

train_X, test_X, train_y, test_y = train_test_split(Fx, Fy, test_size=0.25)

# 5.调用LightGBM模型，使用训练集数据进行训练（拟合）
# Add verbosity=2 to print messages while running boosting
my_model = lgb.LGBMRegressor(objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=20, verbosity=2)
my_model.fit(train_X, train_y, verbose=False)

predictions = my_model.predict(test_X)

# 7.对模型的预测结果进行评判（平均绝对误差）
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
