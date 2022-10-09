import random
import sys

import numpy as np
import pandas


def euclidean_distance(d1, d2):
    ans = 0
    for i in range(len(d1)):
        ans += (np.power(d2[i] - d1[i], 2))
    return np.sqrt(ans)


sums = {}
counts = {}


def mdc(test, results_test):
    accurate = 0
    print('PREDICTED\tACTUAL\tCORRECT?')
    for i, t in enumerate(test):
        distance = -1
        prediction = ''
        for key in sums.keys():
            means = [sums[key][x] / counts[key] for x in range(len(t))]
            dd = euclidean_distance(t, means)
            if (distance == -1) or (distance > dd):
                distance = dd
                prediction = key
            elif distance == dd:
                prediction = random.choice([prediction, key])
        print(prediction, '\t', results_test[i], '\t', prediction == results_test[i])
        accurate += (prediction == results_test[i])
        # update sum and count
        for j in range(len(t)):
            sums[prediction][j] += t[j]
        counts[prediction] += 1
    print(accurate, ' correctly predicted out of ', len(test), ' test samples')
    print('Accuracy = ', (100 * accurate / len(test)), '%')


def calc_mean(training, results_training):
    for i, d in enumerate(training):
        if results_training[i] not in sums:
            ll=[]
            for j in range(len(d)):
                ll.append(d[j])
            sums[results_training[i]]=ll
            counts[results_training[i]] = 1
        else:
            for j in range(len(d)):
                sums[results_training[i]][j] += d[j]
            counts[results_training[i]] += 1


if __name__ == '__main__':
    if len(sys.argv)>1:
        file = sys.argv[1]
    else:
        file= 'IRIS.csv'
    dataset = pandas.read_csv(file)
    # split into training and test data
    split = int(input('Enter the value for splitting: (e.g. if you want to split the data as 75-25 (training-test), '
                      'enter 75)\n'))
    if split == 0:
        print('Sorry..we need at least some training samples')
        exit()
    elif split == 100:
        print('What use is training a model if you do not want to test it??')
        exit()
    elif split < 0 or split > 100:
        print('Invalid input')
        exit()
    training = []
    results_training = []
    test = []
    results_test = []
    for i in range(len(dataset)):
        row = []
        result = ''
        for j, col in enumerate(dataset):
            if j < (dataset.shape[1] - 1):
                row.append(dataset[col][i])
            else:
                result = dataset[col][i]
        if random.random() > (split / 100):
            test.append(row)
            results_test.append(result)
        else:
            training.append(row)
            results_training.append(result)
    # print('Training Data:')
    # print(training)
    # print('Test Data:')
    # print(test)
    calc_mean(training, results_training)
    print('Results:')
    mdc(test, results_test)
