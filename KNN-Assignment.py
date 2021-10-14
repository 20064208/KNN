import random
from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce
from math import sqrt
import numpy

g_dataset = {}
g_test_good = {}
g_test_bad = {}
NUM_ROWS = 32
NUM_COLS = 32
DATA_TRAINING = 'digit-training.txt'
DATA_TESTING = 'digit-testing.txt'
DATA_PREDICT = 'digit-predict.txt'

# kNN parameter
KNN_NEIGHBOR = 7

##########################
##### Load Data  #########
##########################

# Convert next digit from input file as a vector 
# Return (digit, vector) or (-1, '') on end of file
def read_digit(p_fp):
    # read entire digit (inlude linefeeds)
    bits = p_fp.read(NUM_ROWS * (NUM_COLS + 1))
    if bits == '':
        return -1,bits
    # convert bit string as digit vector
    vec = [int(b) for b in bits if b != '\n']
    val = int(p_fp.readline())
    print(val,vec,'val####vec')
    return val,vec

# Parse all digits from training file
# and store all digits (as vectors) 
# in dictionary g_dataset
def load_data(p_filename=DATA_TRAINING):
    global g_dataset
    # Initial each key as empty list 
    g_dataset = defaultdict(list)
    with open(p_filename) as f:
        while True:
            val,vec = read_digit(f)
            if val == -1:
                break
            g_dataset[val].append(vec)

            

##########################
##### kNN Models #########
##########################

# Given a digit vector, returns
# the k nearest neighbor by vector distance
def knn(p_v, size=KNN_NEIGHBOR):
    nn = []
    for d,vectors in g_dataset.items():
        for v in vectors:
            dist = round(distance(p_v,v),2)
            nn.append((dist,d))


    # TODO: find the nearest neigbhors
    '''
    p_v应该是predict_vec
    1.排序
    2.取前size个
    3.return nn
    '''
    nn=sorted(nn,key=lambda item:item[0])
    nn2=nn[0:size]
    return nn2

# Based on the knn Model (nearest neighhor),
# return the target value
def knn_by_most_common(p_v,size=KNN_NEIGHBOR):
    nn = knn(p_v)

    # TODO: target value
    '''
    nn=[(dist,digit),...]
    1.加权平均
    2.确定结果
    '''
    target_set=defaultdict(int)
    dist_sum=sum([ item[0] for item in nn ])
    for dist,d in nn:
        target_set[d]+=(1-dist/dist_sum)
    target_set_2={ dist:d for d,dist in target_set.items()}
    target_max=max(target_set_2.keys())
    target=target_set_2[target_max]
    return target

##########################
##### Prediction  ########
##########################

# Make prediction based on kNN model
# Parse each digit from the predict file
# and print the predicted balue
def predict(p_filename=DATA_PREDICT):
    # TODO
    print('TO DO: show results of prediction')
    '''
    1.读数据
    2.每一个digit进行KNN
    3.出结果
    （4.打印时间）
    ''' 
    ##data_by_random()
    with open(p_filename) as f:
        while True:
            val,vec = read_digit(f)
            if val == -1:
                break
            result=knn_by_most_common(vec)
            print(result)
    
    
    
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
        
    start=datetime.now()

    # TODO: Validate your kNN model with 
    # digits from test file.
    '''
    1.读取test文件数据
    2.验证knn
    3.记录good和bad
    4.show_test打印准确率
    '''
    #为了每个数字量平衡
    data_by_random(150)
    test_set=defaultdict(list)
    with open(p_filename) as f:
        while True:
            val,vec = read_digit(f)
            if val == -1:
                break
            test_set[val].append(vec)
        l=sum([ len(v) for v in test_set.values() ])
        i=0
        print('TODO: Training Info')
        for val,vecs in test_set.items():
            for vec in vecs:
                test_result=knn_by_most_common(vec)
                if test_result==val:
                    g_test_good[val]+=1
                else:
                    g_test_bad[val]+=1
                i+=1
                print("\r进度：{}%".format(round(100*i/l,2)),end='')
        print()
            
    stop=datetime.now()
    show_test(start, stop)

##########################
##### Data Models ########
##########################

# Randomly select X samples for each digit
def data_by_random(size=25):
    for digit in g_dataset.keys():
        g_dataset[digit] = random.sample(g_dataset[digit],size)

##########################
##### Vector     #########
##########################

# Return distance between vectors v & w
'''numpy_array=numpy.array
numpy_square=numpy.square
numpy_sum=numpy.sum
numpy_sqrt=numpy.sqrt'''
def distance(v, w):
    #dist = numpy_sqrt(numpy_sum(numpy_square(numpy_array(v) - numpy_array(w))))
    dist = numpy.sqrt(numpy.sum(numpy.square(numpy.array(v) - numpy.array(w))))
    #a=[ (v[i]-w[i])**2 for i in range(len(v)) ]
    #dist=sqrt(sum(a))
    return dist

##########################
##### Report     #########
##########################

# Show info for training data set
def show_info():
    #print('TODO: Training Info')
    for d in range(10):
        print(d, '=', len(g_dataset[d]))

# Show test results
def show_test(start="????", stop="????"):
    print('Beginning of Validation @ ', start)    
    print('TODO: Testing Info')
    txt=['K={}\n'.format(KNN_NEIGHBOR)]
    a=0    
    for d in range(10):
        good = g_test_good[d]
        bad = g_test_bad[d]
        strs="{}={} {} {}%\n".format(d,good,bad,round(100*good/(good+bad),2))
        txt.append(strs)
        a+=100*good/(good+bad)
        print(d, '=', good, bad, round(100*good/(good+bad),2))
    print('End of Validation @ ', stop)  
    aa="aveage:{}%\n".format(round(a/10,2))
    txt.append(aa)
    with open('K.txt','a') as f:
        f.writelines(txt)

if __name__ == '__main__':
    load_data()
    show_info()
    validate()
    predict()