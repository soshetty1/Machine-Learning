from sklearn import datasets
#imports OneVsRestClassifier class from sklearn.multiclass module, which is a multiclass/multilabel
# strategy for classifying instances into multiple categories.
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report

# Load the breast cancer dataset
cancer = datasets.load_breast_cancer()
#The data and target attributes of the cancer variable are stored in X and y respectively.
X = cancer.data
y = cancer.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train an SVM classifier using OvA, An SVM classifier is trained using the OneVsRestClassifier wrapper
# with a linear kernel, cost parameter (C) of 1, and a random_state set to 0. The classifier is trained on the X_train and y_train data.
clf = OneVsRestClassifier(SVC(kernel='linear', C=1, random_state=0))
clf.fit(X_train, y_train)

# The classifier makes predictions on the X_test data and stores it in the variable y_pred.
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier,
# The accuracy of the classifier is calculated and printed using accuracy_score function.
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

#The function plot_confusion_matrix is defined which plots the confusion matrix of the classifier.
# It takes in the confusion matrix, class labels, normalization flag, title, and color map as input arguments.
#cm represents the confusion matrix, which is a table used to evaluate the performance of a classifier.
# It consists of count of true positive, false positive, true negative, and false negative for each class.
#classes is an array of class labels, used for mapping the class labels to the indices of the confusion matrix.

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
#plt.imshow function from matplotlib is used to display the confusion matrix as a color-coded image,
    # with the title specified by title and the color map specified by cmap.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute the confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=cancer.target_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=cancer.target_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
print(classification_report(y_test, y_pred))
