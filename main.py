import pandas as pd
from collections import defaultdict
import sys
from operator import itemgetter
Dir = '/Users/su/Desktop/python_project/ml-20m/ratings.csv'
all_ratings = pd.read_csv(Dir)
# 时间戳转换
all_ratings['timestamp'] = pd.to_datetime(all_ratings['timestamp'], unit='s')
# bool类型数组
all_ratings["favorable"] = all_ratings['rating'] > 3
# 选取前200作为训练集
# isin是判断矢量化集合的成员资格，返回bool值,以下返回符合资格的成员，返回前200（符合）
ratings = all_ratings[all_ratings['userId'].isin(range(200))]
# 返回一个打分全部高于3.0的dataframe
favorable_ratings = ratings[ratings['favorable']]
# 分组运算 参数是分组键值，书P268
# 书P268
# favorable_ratings.groupby("userId")["movieId"]是
# favorable_ratings['movieId'].groupby(favorable_ratings["userId"])的语法糖
# k是分组名（用户名），v是数据块（电影id）
# frozenset是不可变集合
favorable_reviews_by_users = dict((k, frozenset(v.values))
                                      for k, v in favorable_ratings.groupby("userId")["movieId"])
# 获取影迷数量求和,对每部电影进行支持度计数（）
num_favorable_by_movie = ratings[["movieId", "favorable"]].groupby("movieId").sum()

# Apriori算法实现
frequent_itemsets = {}  # 频繁项集
min_support = 50    # 最小支持度
# 生成初始频繁项集，1维最大项目集L1
frequent_itemsets[1] = dict((frozenset((movie_id,)), row['favorable'])
                            for movie_id, row in num_favorable_by_movie.iterrows()
                            if row["favorable"] > 50)


# k_1_itemsets是一维集
def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    # 生成了一个默认为0的带key的数据字典,values的值具有默认值，keys值自定。
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            # 判断子集，a.issubset(b), a是否是b的子集去，表示用户已经为该电影打分
            if itemset.issubset(reviews):
                                            # 差集，相对补集
                for other_reviewed_movie in reviews - itemset:
                    # 生成超集                      # 合集
                    # 生成候选项集,连接步
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    # 支持度计算
                    # 判断集合是否在超集中，在就+1（剪枝？）
                    counts[current_superset] += 1

    return dict([(itemset, frequency)
                for itemset, frequency in counts.items() if frequency >= min_support])


for k in range(2, 20):
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users,
                                                   frequent_itemsets[k-1],
                                                   min_support=min_support)
    frequent_itemsets[k] = cur_frequent_itemsets
del frequent_itemsets[1]
print(frequent_itemsets)


candidate_rules = []    # 关联规则
for itemset_length, itemset_counts in frequent_itemsets.items():
    # 遍历在频繁项集中出现的每一部电影
    for itemset in itemset_counts.keys():   # 获取项集
        for conclusion in itemset:  # 遍历项集内数据
            premise = itemset - set((conclusion,))
            # 使用前提和结论作为规则
            # 比如 (frozenset({47, 50, 318, 593}), 296)
            #     (frozenset({50}), 47)
            # 数据表示用户喜欢集合内电影，可能会喜欢后一个电影
            candidate_rules.append((premise, conclusion))

# 开始计算置信度

# 存储规则应验的次数
corrent_couts = defaultdict(int)
# 不应验的次数
incorrect_couts = defaultdict(int)

for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        # 判断是否喜欢前提电影
        if premise.issubset(reviews):
            # 判断是否喜欢结论电影
            if conclusion in reviews:
                corrent_couts[candidate_rule] += 1
            else:
                incorrect_couts[candidate_rule] += 1
            # 置信度计算 P(B|A) = P(AB) / P(A)，每条规则的置信度
        rule_confidence = {candidate_rule: corrent_couts[candidate_rule] / float(corrent_couts[candidate_rule]
                                                                                 + incorrect_couts[candidate_rule])
                           for candidate_rule in candidate_rules}
print(rule_confidence.items())


