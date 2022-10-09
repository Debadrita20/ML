import random
import sys

import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier


def euclidean_distance(d1, d2):
    ans=0
    for i in range(len(d1)):
        ans+=(np.power(d2[i]-d1[i],2))
    return np.sqrt(ans)


def knn(training, results, test, actual, k=5):
    dist = np.zeros(len(training))
    accurate = 0
    print('PREDICTED\tACTUAL\tCORRECT?')
    for i in range(len(test)):
        for j in range(len(training)):
            dist[j] = euclidean_distance(test[i], training[j])
        ind = np.argsort(dist)[:k]
        # print(dist)
        # print(ind)
        values=[results[x] for x in ind]
        distances=[dist[x] for x in ind]
        freq = dict()
        mindist=dict()
        for no,val in enumerate(values):
            if val in freq:
                freq[val] += 1
                if mindist[val]>distances[no]:
                    mindist[val]=distances[no]
            else:
                freq[val] = 1
                mindist[val]=distances[no]
        max_count = max(freq.values())
        prediction=''
        for val, count in freq.items():
            if count == max_count:
                if prediction=='':
                    prediction=val
                elif mindist[val]<mindist[prediction]:
                    prediction=val
                elif mindist[val]==mindist[prediction]:
                    prediction=random.choice([val,prediction])
        print(prediction, '\t', actual[i],'\t',prediction == actual[i])
        accurate += (prediction == actual[i])
    print(accurate, ' correctly predicted out of ', len(test), ' test samples')
    print('Accuracy = ',(100*accurate/len(test)),'%')
    return accurate


if __name__ == '__main__':
    k = int(input('Enter the value of K: '))
    if k<=0:
        print('Invalid input')
        exit()
    if len(sys.argv)>1:
        file = sys.argv[1]
    else:
        file= 'IRIS.csv'
    dataset = pandas.read_csv(file)
    # split into training and test data
    split=int(input('Enter the value for splitting: (e.g. if you want to split the data as 75-25 (training-test), '
                    'enter 75)\n'))
    if split==0:
        print('Sorry..we need at least some training samples')
        exit()
    elif split==100:
        print('What use is training a model if you do not want to test it??')
        exit()
    elif split<0 or split>100:
        print('Invalid input')
        exit()
    training = []
    results_training=[]
    test = []
    results_test=[]
    for i in range(len(dataset)):
        row=[]
        result=''
        for j,col in enumerate(dataset):
            if j < (dataset.shape[1] - 1):
                #print(dataset[col][i])
                row.append(float(dataset[col][i]))
            else:
                result=dataset[col][i]
        if random.random() > (split/100):
            test.append(row)
            results_test.append(result)
        else:
            training.append(row)
            results_training.append(result)
    #print('Training Data:')
    #print(training)
    #print('Test Data:')
    #print(test)
    print('Results from own algorithm:')
    acc=knn(training,results_training,test,results_test,k)
    print('Results from scikit-learn algorithm:')
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(training, results_training)
    sk_predictions = knn.predict(test)
    accurate = 0
    print('PREDICTED\tACTUAL\tCORRECT?')
    for i in range(len(results_test)):
        print(sk_predictions[i],'\t',results_test[i],'\t',sk_predictions[i]==results_test[i])
        accurate+=(sk_predictions[i]==results_test[i])
    print(accurate, ' correctly predicted out of ', len(test), ' test samples')
    print('Accuracy = ', (100*accurate / len(test)), '%')
    if acc>accurate:
        print('My algorithm is giving better accuracy')
    elif acc==accurate:
        print('Both algorithms giving same accuracy')
    else:
        print('Scikit learn algorithm is giving better accuracy')
