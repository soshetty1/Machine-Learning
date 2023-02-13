import sklearn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load the breast cancer dataset
cancer = datasets.load_breast_cancer()
#print(cancer.feature_names)
#print(cancer.target_names)

# x contains the feature data and y contains the target labels
x = cancer.data
y = cancer.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#print(x_train, y_train)
classes = ['malignant' 'benign']
# creates the svm classifier object 'clf' , c is used to set the margin value, Define the SVM classifier
clf = svm.SVC(kernel="linear", C=2)

# Perform 10-fold cross validation
scores = cross_val_score(clf, x, y, cv=10)
# Calculate the mean and standard deviation of the scores
mean_score = np.mean(scores)
std_dev = np.std(scores)

# Print the average accuracy score and the standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (mean_score, std_dev * 2))

#clf = KNeighborsClassifier(n_neighbors=7)
# Train the SVM classifier on the training data
clf.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)

# Get the indices of the samples where y_test and y_pred are different
diff_idx = np.where(y_test != y_pred)[0]

# Plotting the graph, s= The size of the markers
plt.scatter(np.arange(len(y_test)), y_test, c='b', label='True Values', s=25)
plt.scatter(np.arange(len(y_pred)), y_pred, c='r', label='Predicted Values', marker='x', s=50)

# Highlight the misclassified samples
plt.scatter(diff_idx, y_test[diff_idx], c='g', label='Misclassified', s=25, alpha=0.5)

plt.xlabel('Samples')
plt.ylabel('Classes')
plt.title('Accuracy: {:.2f}'.format(mean_score))
plt.legend()
plt.show()
