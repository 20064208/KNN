import random
from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce

g_dataset = {}
g_test_good = {}
g_test_bad = {}
NUM_ROWS = 32
NUM_COLS = 32
DATA_TRAINING = 'digit-training.txt'
DATA_TESTING = 'digit-testing.txt'
DATA_PREDICT = 'digit-predict.txt'

# kNN parameter
KNN_NEIGHBOR = 7  # knn邻居  选取与当前点距离最小的7个点


##########################
##### Load Data  #########
##########################

# Convert next digit from input file as a vector  将输入文件中的下一个数字转换为矢量
# Return (digit, vector) or (-1, '') on end of  在文件末尾返回（数字、向量）或（-1，'）
def read_digit(p_fp):
    # read entire digit (inlude linefeeds)
    bits = p_fp.read(NUM_ROWS * (NUM_COLS + 1))
    if bits == '':
        return -1, bits
    # convert bit string as digit vector
    vec = [int(b) for b in bits if b != '\n']
    val = int(p_fp.readline())
    return val, vec


# Parse all digits from training file 解析训练文件中的所有数字
# and store all digits (as vectors)  #并存储所有数字（作为向量）
# in dictionary g_dataset 字典中的g_数据集
def load_data(p_filename=DATA_TRAINING):
    global g_dataset
    # Initial each key as empty list 
    g_dataset = defaultdict(list)
    with open(p_filename) as f:
        while True:
            val, vec = read_digit(f)
            if val == -1:
                break
            g_dataset[val].append(vec)


##########################
##### kNN Models #########
##########################

# Given a digit vector, returns    #给定一个数字向量，返回向量距离的k近邻要有7个近邻
# the k nearest neighbor by vector distance
def knn(p_v, size=KNN_NEIGHBOR):
    nn = []
    for d, vectors in g_dataset.items():
        for v in vectors:
            dist = round(distance(p_v, v), 2)
            nn.append((dist, d))
    print(nn)

    # TODO: find the nearest neigbhors
    # nn = sorted(nn, key=lambda item: item[0])
    # nn2 = nn[0:size]
    # print(nn2)
    # return nearest_nei


# Based on the knn Model (nearest neighhor),
# return the target value
# 基于knn模型（最近的Neighor），返回目标值
def knn_by_most_common(p_v):
    nn = knn(p_v)

    # # TODO: target value
    # target_set = defaultdict(int)
    # dist_sum = sum([item[0] for item in nn])
    # for dist, d in nn:
    #     target_set[d] += (1 - dist / dist_sum)
    # target_set_2 = {dist: d for d, dist in target_set.items()}
    # target_max = max(target_set_2.keys())
    # target = target_set_2[target_max]
    # print(target)
    # return target


##########################
##### Prediction  ########
##########################

# Make prediction based on kNN model
# Parse each digit from the predict file
# and print the predicted balue
def predict(p_filename=DATA_PREDICT):
    # TODO
    print('TO DO: show results of prediction')


##########################
##### Accuracy   #########
##########################

# Compile an accuracy report by
# comparing the data set with every
# digit from the testing file 
def validate(p_filename=DATA_TESTING):
    global g_test_bad, g_test_good
    g_test_bad = defaultdict(int)
    g_test_good = defaultdict(int)

    start = datetime.now()

    # TODO: Validate your kNN model with 
    # digits from test file.

    stop = datetime.now()
    show_test(start, stop)


##########################
##### Data Models ########
##########################

# Randomly select X samples for each digit
def data_by_random(size=25):
    for digit in g_dataset.keys():
        g_dataset[digit] = random.sample(g_dataset[digit], size)


##########################
##### Vector     #########
##########################

# Return distance between vectors v & w
def distance(v, w):
    return 0


##########################
##### Report     #########
##########################

# Show info for training data set
def show_info():
    print('TODO: Training Info')
    for d in range(10):
        print(d, '=', len(g_dataset[d]))


# Show test results
def show_test(start="????", stop="????"):
    print('Beginning of Validation @ ', start)
    print('TODO: Testing Info')
    for d in range(10):
        good = g_test_good[d]
        bad = g_test_bad[d]
        print(d, '=', good, bad)
    print('End of Validation @ ', stop)


if __name__ == '__main__':
    load_data()
    show_info()
    validate()
    predict()
