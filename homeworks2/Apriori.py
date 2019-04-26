# encoding: utf-8
import pandas as pd
import os
import json

class Data():
    def __init__(self):
        self.dataPath = './wine-reviews/'
        # self.srcData = pd.read_csv(self.dataPath + 'new_winemag-data_first150k.csv')
        self.srcData = pd.read_csv(self.dataPath + 'new_winemag-data-130k-v2.csv')
        self.dataTuple = []
        self.minSup = 0.25  # 最小支持度
        self.minConf = 0.5  # 最小置信度

    def set_dataTuple(self):
        # 提取预处理后的数据
        content = self.srcData
        columns = []
        for title in content.columns.values:
            feature = [title] + list(content[title])
            columns.append(feature)
        rows = list(zip(*columns))
        featureNames = rows[0]
        for data_line in rows[1:]:
            data_line_set = []
            for i, value in enumerate(data_line):
                data_line_set.append((featureNames[i], value))
            self.dataTuple.append(data_line_set)
        # print(self.dataTuple[0:10])

    def genarateOneFrequenceSet(self):
        # 生成1-频繁项目集
        # 存储1-频繁项目集：（标称名，取值）
        oneFreSet = []
        for data in self.dataTuple:
            for item in data:
                if [item] not in oneFreSet:
                    oneFreSet.append([item])
        oneFreSet.sort()
        return [frozenset(item) for item in oneFreSet]

    def filterData(self, dataSet, FreSet):
        # 过滤函数
        # 统计数据集dataTuple中1-频繁项目集中各个元素的出现频率,过滤掉低于最小支持度的项集
        # 返回大于最小支持度的集合，以及支持度
        oneFreSetCount = dict()
        for data in dataSet:
            for cand in FreSet:
                if cand.issubset(data):
                    if cand not in oneFreSetCount:
                        oneFreSetCount[cand] = 1
                    else:
                        oneFreSetCount[cand] += 1

        num_items = float(len(dataSet))
        listSuitData = []
        dataSup = dict()
        # 过滤非频繁项集
        for key in oneFreSetCount:
            support = oneFreSetCount[key] / num_items
            if support >= self.minSup:
                listSuitData.insert(0, key)
            dataSup[key] = support
        return listSuitData, dataSup

    def judgeItem(self, dataSet, k):
        # 当待选项集不是单个元素时， 是否只有最后一项不同
        return_list = []
        len_dataSet = len(dataSet)

        for i in range(len_dataSet):
            for j in range(i+1, len_dataSet):
                # 第k-2个项相同时，将两个集合合并
                L1 = list(dataSet[i])[:k-2]
                L2 = list(dataSet[j])[:k-2]
                L1.sort()
                L2.sort()
                if L1 == L2:
                    return_list.append(dataSet[i] | dataSet[j])
        return return_list

    def apriori(self):
        self.set_dataTuple()
        oneFreSet = self.genarateOneFrequenceSet()
        dataSet = [set(data) for data in self.dataTuple]
        listSuitData, dataSup = self.filterData(dataSet, oneFreSet)
        listData = [listSuitData]

        layer = 2
        while len(listData[layer - 2]):
            Ck = self.judgeItem(listData[layer - 2], layer)
            Lk, support_k = self.filterData(dataSet, Ck)
            dataSup.update(support_k)
            listData.append(Lk)
            layer += 1
        return listData, dataSup

    def generate_rules(self, L, support_data):
        big_rules_list = []
        for i in range(1, len(L)):
            for freq_set in L[i]:
                H1 = [frozenset([item]) for item in freq_set]
                # 只获取有两个或更多元素的集合
                if i > 1:
                    self.rules_from_conseq(freq_set, H1, support_data, big_rules_list)
                else:
                    self.cal_conf(freq_set, H1, support_data, big_rules_list)
        return big_rules_list

    def rules_from_conseq(self, freq_set, H, support_data, big_rules_list):
        # H->出现在规则右部的元素列表
        m = len(H[0])
        if len(freq_set) > (m + 1):
            Hmp1 = self.judgeItem(H, m + 1)
            Hmp1 = self.cal_conf(freq_set, Hmp1, support_data, big_rules_list)
            if len(Hmp1) > 1:
                self.rules_from_conseq(freq_set, Hmp1, support_data, big_rules_list)

    def cal_conf(self, freq_set, H, support_data, big_rules_list):
        # 评估生成的规则
        prunedH = []
        for conseq in H:
            sup = support_data[freq_set]
            conf = sup / support_data[freq_set - conseq]
            lift = conf / support_data[conseq]
            consine = sup / ((support_data[conseq] * support_data[freq_set - conseq]) ** 0.5)
            if conf >= self.minConf:
                big_rules_list.append((freq_set - conseq, conseq, sup, conf, lift, consine))
                prunedH.append(conseq)
        return prunedH

    def association(self):
        # 获取频繁项集
        freq_set, support_data = self.apriori()
        # 将频繁项集输出到结果文件
        freq_set_file = open(self.dataPath + 'frequent_set.csv', 'w')
        support_data_out = sorted(support_data.items(), key= lambda d:d[1],reverse=True)
        # 以support降序排列
        dataWrit = []
        for (key, value) in support_data_out:
            set_result = list(key)
            dataWrit.append((set_result[0][0],set_result[0][1],value))
        _data = pd.DataFrame(dataWrit)
        # print(_data)
        csv_headers = ['attribute', 'attributeValue','sup']
        _data.to_csv(freq_set_file, header=csv_headers, index=False, mode='a+', encoding='utf-8')
        freq_set_file.close()

        # 获取强关联规则列表
        big_rules_list = self.generate_rules(freq_set, support_data)
        big_rules_list = sorted(big_rules_list, key= lambda x:x[3], reverse=True)
        # 将关联规则输出到结果文件
        rules_file = open(self.dataPath + 'rules.csv', 'w')
        dataRulesWrit = []
        for result in big_rules_list:
            X_set, Y_set, sup, conf, lift, consine = result
            pre = list(X_set)
            post = list(Y_set)
            dataRulesWrit.append((pre[0][0],pre[0][1],post[0][0],post[0][1],sup,conf,lift,consine))
        _dataRules = pd.DataFrame(dataRulesWrit)
        # print(_data)
        csv_headers = ['frontAtrri', 'frontValue', 'backAttri','backValue','sup','conf','lift','consine']
        _dataRules.to_csv(rules_file, header=csv_headers, index=False, mode='a+', encoding='utf-8')
        rules_file.close()

if __name__ == '__main__':
    data = Data()
    data.association()
    # print(data.genarateOneFrequenceSet()[1:10])