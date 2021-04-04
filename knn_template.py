import random
from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce
import math
from tqdm import tqdm

g_dataset = {}
g_test_good = {}
g_test_bad = {}
g_test = {}
NUM_ROWS = 32
NUM_COLS = 32
DATA_TRAINING = 'digit-training.txt'
DATA_TESTING = 'digit-testing.txt'
DATA_PREDICT = 'digit-predict.txt'

# kNN parameter
KNN_NEIGHBOR = 100


# Convert next digit from input file as a vector
# Return (digit, vector) or (-1, '') on end of file
def read_digit(p_fp):
    # read entire digit (inlude linefeeds)
    bits = p_fp.read(NUM_ROWS * (NUM_COLS + 1))
    if bits == '':
        return -1, bits
    # convert bit string as digit vector
    vec = [int(b) for b in bits if b != '\n']
    val = int(p_fp.readline())
    return val, vec


# Parse all digits from training file
# and store all digits (as vectors) 
# in dictionary g_dataset 
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


# Given a digit vector, returns
# the k nearest neighbor by vector distance
def knn(p_v, size=KNN_NEIGHBOR):
    nn = []
    for d, vectors in g_dataset.items():
        for v in vectors:  # v:list
            dist = round(distance(p_v, v), 2)  # distance
            nn.append((dist, d))  # (distance, value)
    nn.sort(key=lambda x: x[0])  # sort by distance
    # TODO: find the nearest neigbhors
    # print(nn[:100])   return :size
    return nn[:size]


# Based on the knn Model (nearest neighhor),
# return the target value
def knn_by_most_common(p_v):
    nn = knn(p_v)
    print(nn)
    # TODO: target value
    return Counter([v[1] for v in nn]).most_common(1)  # return most commmon


# Make prediction based on kNN model
# Parse each digit from the predict file
# and print the predicted balue
def predict(p_filename=DATA_PREDICT):
    # TODO
    print('TO DO: show results of prediction')
    vecT = []
    with open(p_filename) as p:
        while True:
            val, vec = read_digit(p)
            if val == -1:
                break
            vecT.append(vec)
    for v in vecT:
        print(knn_by_most_common(v)[0][0], "%d%%" % knn_by_most_common(v)[0][1])


# Compile an accuracy report by
# comparing the data set with every
# digit from the testing file 
def validate(p_filename=DATA_TESTING):
    global g_test_bad, g_test_good, g_test
    g_test_bad = defaultdict(int)
    g_test_good = defaultdict(int)
    g_test = defaultdict(list)

    start = datetime.now()

    # TODO: Validate your kNN model with 
    # digits from test file.
    with open(p_filename) as p:
        while True:
            val, vec = read_digit(p)
            if val == -1:
                break
            g_test[val].append(vec)
    for val, vecs in g_test.items():
        for v in tqdm(vecs):
            valPre, numT = knn_by_most_common(v)[0]
            g_test_good[val] = numT
            g_test_bad[val] = KNN_NEIGHBOR - numT

    stop = datetime.now()
    show_test(str(start), str(stop))


# Randomly select X samples for each digit
def data_by_random(size=25):
    for digit in g_dataset.keys():
        g_dataset[digit] = random.sample(g_dataset[digit], size)


# Return distance between vectors v & w
def distance(v, w):
    return reduce(lambda x, y: x + y, [round(math.sqrt((vi - wi) ** 2)) for vi, wi in zip(v, w)])


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
    # show_info()
    # validate()
    # show_test()
    predict()
