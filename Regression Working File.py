import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# each data is separated by ; in csv file
data = pd.read_csv("student-mat.csv", sep=";")
#prints the first 5 elements of our dataframe
print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures","absences"]]
#G3 is final grade
predict = "G3"
 # this drops g3 from the data frame
X = np.array(data.drop(labels=[predict], axis=1))
#here we are taking the g3 value
Y = np.array(data[predict])
#x_train and y-train will have sextion of  X and Y respectively . x_test,y_test is used to test the accuracy of our model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

"""
best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
#create model
    linear = linear_model.LinearRegression()
#This is a method that is used to train a linear regression model in Python using scikit-learn library.fit() method is used to train the model on the training data
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
       best = accuracy
       with open("studentmodel.pickle", "wb") as f:
        pickle.dump(linear, f) """

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)


print("Coefficient: \n", linear.coef_)
print("intercept: \n", linear.intercept_ )

#using the trained linear regression model to make predictions on the test data.
#linear is an instance of the LinearRegression class from scikit-learn library that has been trained on the training data using the fit() method.
#The predict() method is used to make predictions on new data, which is passed as the argument x_test and returns an array of shape (n_samples,) that contains the predicted target values for the test data.
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()