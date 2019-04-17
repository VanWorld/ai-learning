from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import shift
from sklearn.metrics import accuracy_score


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


def shift_train_set(train_set, shifts):
    """
    效率太低
    :param train_set:
    :param shifts:
    :return:
    """
    result = shift(train_set[0].reshape((28, 28)), shifts, cval=0).reshape((28*28))
    for row in train_set[1:]:
        row_new = shift(row.reshape((28, 28)), shifts, cval=0).reshape((28*28))
        result = np.vstack([result, row_new])
    return result


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dx, dy], cval=0, mode="constant")
    return shifted_image.reshape([-1])


if __name__ == "__main__":
    mnist = fetch_mldata('MNIST original')
    X, y = mnist['data'], mnist['target']
    some_digit = X[36000]
    # print(X.shape, y.shape)
    # print(X)
    # print(y)
    plt.imshow(X[36000].reshape((28, 28)), cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis('off')
    plt.show()

    # split train set and test set
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # shuffle the train set
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    # 5 detector
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    print(sgd_clf.predict([some_digit]))

    result = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    print(result)

    # never 5 dectector
    never_5_clf = Never5Classifier()
    print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

    # confusion matrix
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    conf_matrix = confusion_matrix(y_train_5, y_train_pred)
    print(type(conf_matrix), conf_matrix)

    print(precision_score(y_train_5, y_train_pred), recall_score(y_train_5, y_train_pred))

    # manual threshold
    y_scores = sgd_clf.decision_function([some_digit])
    print(y_scores)
    threshold_1 = 0
    y_some_digit_pred_1 = (y_scores > threshold_1)

    threshold_2 = 200000
    y_some_digit_pred_2 = (y_scores > threshold_2)
    print(y_some_digit_pred_1, y_some_digit_pred_2)

    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    # plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    # roc curve
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    # plot_roc_curve(fpr, tpr, thresholds)
    # plt.show()

    # multiclass classifier
    sgd_clf.fit(X_train, y_train)
    print(sgd_clf.predict([some_digit]))
    some_digit_scores = sgd_clf.decision_function([some_digit])
    print(some_digit_scores)

    np.argmax(some_digit_scores)
    print(sgd_clf.classes_)

    # one vs one
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
    ovo_clf.fit(X_train, y_train)
    print(ovo_clf.predict([some_digit]))
    len(ovo_clf.estimators_)

    # scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_max = confusion_matrix(y_train, y_train_pred)
    print(conf_max)
    # plt.matshow(conf_max, cmap=plt.cm.gray)

    row_sum = conf_max.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_max / row_sum
    print(row_sum, norm_conf_mx)
    np.fill_diagonal(norm_conf_mx, 0)
    # plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

    y_train_large = (y_train > 7)
    y_train_odd = (y_train % 2 == 1)
    y_multi_label = np.c_[y_train_large, y_train_odd]

    knn_clf = KNeighborsClassifier()
    # knn_clf.fit(X_train, y_multi_label)
    # print(knn_clf.predict([some_digit]))

    params_grid = [
        {'weights': ['uniform', 'distance'], 'n_neighbors': [3, 4, 5]},
    ]

    # grid_search = GridSearchCV(knn_clf, params_grid, cv=5, verbose=3, n_jobs=-1)
    # grid_search.fit(X_train, y_train)
    # print()
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)

    # exercise 2
    X_train_augmented = [image for image in X_train]
    y_train_augmented = [label for label in y_train]

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for image, label in zip(X_train, y_train):
            X_train_augmented.append(shift_image(image, dx, dy))
            y_train_augmented.append(label)
    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)

    shuffle_idx = np.random.permutation(len(X_train_augmented))
    X_train_augmented = X_train_augmented[shuffle_idx]
    y_train_augmented = y_train_augmented[shuffle_idx]
    print(X_train_augmented.shape)
    print(y_train_augmented.shape)

    best_params = {'n_neighbors': 4, 'weights': 'distance', 'n_jobs': -1}
    knn_clf = KNeighborsClassifier(**best_params)
    knn_clf.fit(X_train_augmented, y_train_augmented)
    y_predict = knn_clf.predict(X_test)
    print(knn_clf.predict([some_digit]))
    print(accuracy_score(y_test, y_predict))