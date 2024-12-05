import numpy as np
import pandas
from sklearn import tree, datasets
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class node:
    def __init__(self, feature=None, branch_values=None, children=None, *, value=None):
        self.feature = feature
        self.branch_values=branch_values
        self.children=children
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeDisc:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _is_finished(self, depth):
        if (depth >= self.max_depth
                or self.n_class_labels == 1
                or self.n_samples < self.min_samples_split):
            return True
        return False

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_label = np.argmax(np.bincount(y))
            return node(value=most_common_label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat = self._best_split(X, y, rnd_feats)
        labels,children = self._create_split(X[:, best_feat])
        children_nodes=[]
        for c in children:
            children_nodes.append(self._build_tree(X[c, :], y[c], depth + 1))
        return node(best_feat, labels, children_nodes)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def _create_split(self, X):
        labels=np.unique(X)
        children=[]
        for label in labels:
            children.append(np.argwhere(X == label).flatten())
        return labels,children

    def _information_gain(self, X, y):
        parent_loss = self._entropy(y)
        labels, children = self._create_split(X)
        n= len(y)
        numbers=[]
        for c in children:
            numbers.append(len(c))
            if len(c)==0:
                return 0
        child_loss=0
        # print(children)
        for i in range(len(children)):
            child_loss += (numbers[i] / n) * self._entropy(y[children[i]])
        return parent_loss - child_loss

    def _gini(self, X, y, thresh):
        pass

    def _best_split(self, X, y, features):
        split = {'score': - 1, 'feat': None}

        for feat in features:
            X_feat = X[:, feat]
            score = self._information_gain(X_feat, y)
            if score > split['score']:
                split['score'] = score
                split['feat'] = feat

        return split['feat']

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        for i in range(len(node.branch_values)):
            if x[node.feature]==node.branch_values[i]:
                return self._traverse_tree(x,node.children[i])

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# data = pandas.read_csv('play_tennis.csv')
data = pandas.read_csv('balance-scale.data')
X, y = [],[]
vals={'B':0,'L':1,'R':2}
for i in range(len(data)):
    row = []
    result = ''
    for j, col in enumerate(data):
        if j==0:
            result = vals[data[col][i]]
        else:
            row.append(data[col][i])
        '''if j < (data.shape[1] - 1):
            row.append(data[col][i])'''
    X.append(row)
    y.append(result)
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.6)
clf = DecisionTreeDisc(max_depth=14)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)
print("Accuracy from own algorithm:", acc*100,'%')
clf_entropy = DecisionTreeClassifier(criterion="entropy",max_depth=14, min_samples_leaf=2)
# Performing training
clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
print ("Accuracy from scikit-learn algorithm: ",accuracy_score(y_test,y_pred)*100,"%")
