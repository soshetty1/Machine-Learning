from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import metrics
# Load the breast cancer dataset
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)


clf = RandomForestClassifier(n_estimators=100)

# Perform 10-fold cross validation
scores = cross_val_score(clf, x, y, cv=10)
# Calculate the mean and standard deviation of the scores
mean_score = np.mean(scores)
std_dev = np.std(scores)

# Print the average accuracy score and the standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (mean_score, std_dev * 2))
clf.fit(x_train, y_train)

# Predict class labels for the test set
y_pred = clf.predict(x_test)

# Compute the accuracy score
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Plot the graph
# Plot the graph
y_test = np.array(y_test)
y_pred = np.array(y_pred)

mask_malignant = (y_test == 0)
mask_benign = (y_test == 1)

plt.scatter(np.arange(len(y_test))[mask_malignant], y_test[mask_malignant], color='blue', label='True Malignant')
plt.scatter(np.arange(len(y_test))[mask_benign], y_test[mask_benign], color='red', label='True Benign')

plt.scatter(np.arange(len(y_pred))[mask_malignant], y_pred[mask_malignant], color='lightblue', marker='x', label='Predicted Malignant')
plt.scatter(np.arange(len(y_pred))[mask_benign], y_pred[mask_benign], color='pink', marker='x', label='Predicted Benign')

plt.xlabel('Samples')
plt.ylabel('Classes')
plt.title('Accuracy: {:.2f}'.format(mean_score))
plt.legend()
plt.show()