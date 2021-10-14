import random
from collections import defaultdict, Counter
from math import sqrt
from datetime import datetime

g_dataset = {}  # 数据集
g_test_good = {}
g_test_bad = {}
NUM_ROWS = 32
NUM_COLS = 32
DATA_TRAINING = 'digit-training.txt'  # 培训
DATA_TESTING = 'digit-testing.txt'  # 测试
DATA_PREDICT = 'digit-predict.txt'  # 显示预测结果

# kNN parameter   （参数）
KNN_NEIGHBOR = 7  # knn邻居  选取与当前点距离最小的7个点


##########################
##### Load Data  #########  加载数据
##########################

# Convert next digit from input file as a vector   将输入文件中的下一个数字转换为矢量
# Return (digit, vector) or (-1, '') on end of file   在文件末尾返回（数字、向量）或（-1，'）
def read_digit(p_fp):
    # read entire digit (inlude linefeeds) #读取整个数字（包括换行符）
    bits = p_fp.read(NUM_ROWS * (NUM_COLS + 1))
    if bits == '':
        return -1, bits
    # convert bit string as digit vector  #将位字符串转换为数字向量
    vec = [int(b) for b in bits if b != '\n']  # 数
    val = int(p_fp.readline())  # 行
    return val, vec


# Parse all digits from training file  #解析训练文件中的所有数字
# and store all digits (as vectors) #并存储所有数字（作为向量）
# in dictionary g_dataset   #字典中的g_数据集
def load_data(p_filename=DATA_TRAINING):
    global g_dataset  # 全局变量数据集
    # Initial each key as empty list #每个键的首字母都是空列表
    g_dataset = defaultdict(list)
    with open(p_filename) as f:  # 打开培训文件
        while True:  # 如果打开了
            val, vec = read_digit(f)  # 调用上面那个将文件里的数字转成向量的函数 返回数字、向量
            # print(val)
            if val == -1:  # 遇到换行符了 就停止
                break
            g_dataset[val].append(vec)  # 数据集里添加数字


##########################
##### kNN Models #########
##########################

# Given a digit vector, returns             #给定一个数字向量，返回
# the k nearest neighbor by vector distance #向量距离的k近邻
def knn(p_v, size=KNN_NEIGHBOR):  # 要有7个近邻
    nn = []  # 定义一个空列表
    for d, vectors in g_dataset.items():  # 数据集中的每行数  数字 值
        for v in vectors:  # v是每行里的数字
            dist = round(distance(p_v, v), 2)  # 算出距离  四舍五入保留两位小数
            nn.append((dist, d))  # （距离，行数值）
    '''
    TODO: find the nearest neigbhors
    待办事项：找到最近的邻居
    使用模块“collections”中的Counter（）排序
    most_common（）返回最近的邻居
    将k个邻居存放在nearest_nei中
    '''
    nn = sorted(nn, key=lambda item: item[0])
    nn2 = nn[0:size]
    return nn2
    # nearest_nei = []
    # for i in range(1, size + 1):
    #     b = Counter(nn).most_common(i)[0][0]
    #     nearest_nei.append(b)
    # # print(nearest_nei)
    # return nearest_nei


# Based on the knn Model (nearest neighhor),
# return the target value
# 基于knn模型（最近的Neighor），返回目标值
def knn_by_most_common(p_v, size=KNN_NEIGHBOR):  # 从knn里找出最常见的
    nn = knn(p_v)  # 上个函数的返回值 nerest_nei
    # TODO: target value   全部：目标值
    target = sum([item[1] for item in nn]) // size
    return target


##########################
##### Prediction  ########
##########################

# Make prediction based on kNN model        #基于kNN模型进行预测
# Parse each digit from the predict file    #解析predict文件中的每个数字
# and print the predicted balue             #并打印预测的balue
def predict(p_filename=DATA_PREDICT):
    global g_dataset  # 全局变量数据集
    # TODO 待办事项：显示预测结果
    print('TO DO: show results of prediction.')
    with open(p_filename) as f:
        while True:
            val, vec = read_digit(f)
            if val == -1:
                break
            prediction = knn_by_most_common(vec)
            print(prediction)

##########################
##### Accuracy   #########   精确
##########################

# Compile an accuracy report by             #编制一份准确度报告
# comparing the data set with every         #将数据集与每个
# digit from the testing file               #测试文件中的数字
def validate(p_filename=DATA_TESTING):
    global g_test_bad, g_test_good
    g_test_bad = defaultdict(int)  # 初始化为0        defaultdict(<class 'int'>, {})
    g_test_good = defaultdict(int)  # 初始化为0         defaultdict(<class 'int'>, {})

    start = datetime.now()  # 开始时间

    # TODO: Validate your kNN model with
    # digits from test file.
    # TODO:用测试文件的数字来验证knn模型 。
    test_set = defaultdict(list) #构建一个默认value为list的字典，
    with open(p_filename) as f:
         while True:
             val, vec = read_digit(f)
             if val == -1:
                 break
             test_set[val].append(vec)
         print('TODO: Training Info')
         for val, vecs in test_set.items():
             for vec in vecs:
                 test_result = knn_by_most_common(vec)  #预测结果
                 if test_result == val:
                     g_test_good[val] += 1
                 else:
                     g_test_bad[val] += 1

    stop = datetime.now()  # 结束时间
    show_test(start, stop)


##########################
##### Data Models ########  数据模型
##########################

# Randomly select X samples for each digit   #为每个数字随机选择X个样本
def data_by_random(size=30):
    for digit in g_dataset.keys():
        g_dataset[digit] = random.sample(g_dataset[digit], size)


##########################
##### Vector     #########  矢量图
##########################

# Return distance between vectors v & w
# 向量v&amp;w之间的返回距离
def distance(v, w):#欧式距离
    distance= [(v[i] - w[i]) ** 2 for i in range(len(v))]
    dist=sqrt(sum(distance))
    return dist


##########################
##### Report     #########  汇报
##########################

# Show info for training data set
# 显示训练数据集的信息
def show_info():
    print('TODO: Training Info')  # 全部：培训信息
    Total_sample = 0
    for d in range(10):
        print(d, '=', len(g_dataset[d]))
        Total_sample += len(g_dataset[d])

    print('------------------------------------')
    print('Total Samples =', Total_sample)
    print('------------------------------------')


# Show test results  #显示测试结果
def show_test(start="????", stop="????"):
    print('Beginning of Validation @ ', start)
    print('TODO: Testing Info')  # 测试信息
    for d in range(10):
        good = g_test_good[d]
        bad = g_test_bad[d]
        print('数字：',d, '预测成功的个数：', good,'预测失败的个数', bad,'正确率为', round(100 * good / (good + bad), 2),'%')
    print('End of Validation @ ', stop)


if __name__ == '__main__':
    load_data()
    show_info()
    validate()
    predict()

