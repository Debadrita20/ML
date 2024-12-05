import numpy as np
import pandas
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
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
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return node(best_feat, best_thresh, left_child, right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        split = {'score': - 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# data = datasets.load_breast_cancer()
# data = datasets.load_iris()
'''data = pandas.read_csv('play_tennis.csv')
X, y = [],[]
for i in range(len(data)):
    row = []
    result = ''
    for j, col in enumerate(data):
        if j==0:
            continue
        if j < (data.shape[1] - 1):
            row.append(data[col][i])
        else:
            result= data[col][i]=='Yes'
    X.append(row)
    y.append(result)'''
data = pandas.read_csv('balance-scale.data')
X, y = [],[]
vals={'B':0,'L':1,'R':2}
list_of_sets=[]
for i in range(len(data)):
    row = []
    result = ''
    for j, col in enumerate(data):
        if i==0:
            list_of_sets.append(set())
        if j==0:
            result = vals[data[col][i]]
        else:
            row.append(data[col][i])
    X.append(row)
    y.append(result)

# X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.5)
clf = DecisionTree(max_depth=12)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)
print("Accuracy from own algorithm:", acc*100,'%')
print('Confusion matrix:\n',confusion_matrix(y_test,y_pred))
clf_entropy = DecisionTreeClassifier(criterion="entropy",max_depth=12, min_samples_leaf=2)
# Performing training
clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
print ("Accuracy from scikit-learn algorithm: ",accuracy_score(y_test,y_pred)*100,"%")
print('Confusion matrix:\n',confusion_matrix(y_test,y_pred))