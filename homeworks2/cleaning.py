# -*- coding:UTF-8 -*-
import pandas as pd

path = 'wine-reviews/'
# srcData = pd.read_csv(path + 'winemag-data_first150k.csv')
srcData = pd.read_csv(path + 'winemag-data-130k-v2.csv')
for colNam in srcData.columns.values:
    # 提取出标称数据，同时删除空值数据
    if srcData[colNam].dtypes == "int64" or srcData[colNam].dtypes == "float64":
        srcData = srcData.drop(columns=[colNam])
resData = srcData.dropna()
#print(srcData.count())
print(resData.count()) # org = 150930, rel=39427
# resData.to_csv(path + 'new_winemag-data_first150k.csv', index=False)
resData.to_csv(path + 'new_winemag-data-130k-v2.csv', index=False)