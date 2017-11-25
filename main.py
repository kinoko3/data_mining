import pandas as pd
from collections import defaultdict
import sys
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
            # 判断子集，a.issubset(b), a是否是b的子集
            if itemset.issubset(reviews):
                                            # 差集，相对补集
                for other_reviewed_movie in reviews - itemset:
                                                # 合集
                    # 生成超集
                    current_superset = itemset | frozenset((other_reviewed_movie,))

                    counts[current_superset] += 1

    return dict([(itemset, frequency)
                for itemset, frequency in counts.items() if frequency >= min_support])


for k in range(2, 20):
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users,
                                                   frequent_itemsets[k-1],
                                                   min_support=min_support)
    frequent_itemsets[k] = cur_frequent_itemsets

    if len(cur_frequent_itemsets) == 0:
        print("什么都没有 {}".format(k))
        sys.stdout.flush()
        break
    else:
        print("{}长度{}".format(len(cur_frequent_itemsets), k))
        sys.stdout.flush()
print(frequent_itemsets)

