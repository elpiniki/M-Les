import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#machine leaning libraries
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

#machine leanring input data transform libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

#machine leanirng validation libraries
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


#function for reading as a dataframe
def datareader(csvfile):
    data = pd.read_csv(csvfile)
    return data

#read the data
winddata = datareader("windDatahr_month.csv")

#split the data into 80% train and 20% test
train_set, test_set = train_test_split(winddata, test_size=0.2, random_state=42)

#keep data you need for each input 
x_train_set = train_set.drop("Energy_kWh", axis=1)
x_test_set = test_set.drop("Energy_kWh", axis=1)
y_train_set = train_set["Energy_kWh"].copy()
y_test_set = test_set["Energy_kWh"].copy()

#normalize the data
scaler = StandardScaler()
scaler.fit(x_train_set)
x_train_set = scaler.transform(x_train_set)
scaler.fit(x_test_set)
x_test_set = scaler.transform(x_test_set)

#linear regression model
lin_reg = LinearRegression()
lin_reg.fit(x_train_set, y_train_set)
y_predictions = lin_reg.predict(x_test_set)

#linear regression scores
lin_mse = mean_squared_error(y_test_set, y_predictions)
lin_rmse = np.sqrt(lin_mse)
print "Mean Squared Error: %s" %lin_rmse
lin_mae = mean_absolute_error(y_test_set, y_predictions)
print "Mean Absolute Error: %s" %lin_mae

#plot results
plt.plot(y_test_set, c='b', label='y_test_set')
plt.plot(y_predictions, c='g', label='y_predictions')
plt.legend(loc='upper left')
plt.xlabel('Data points')
plt.ylabel('Energy (kWh)')
plt.title("Linear Regression")
plt.ylim((-3000, 18000))
plt.xlim((0,150))
plt.savefig("LinPLot")
plt.close()

#Decision Tree model
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(x_train_set, y_train_set)
y_predictions = tree_reg.predict(x_test_set)

#Decision Tree regression scores
tree_mse = mean_squared_error(y_test_set, y_predictions)
tree_rmse = np.sqrt(tree_mse)
print "Mean Squared Error: %s" %tree_rmse
tree_mae = mean_absolute_error(y_test_set, y_predictions)
print "Mean Absolute Error: %s" %tree_mae

#plot results
plt.plot(y_test_set, c='b', label='y_test_set')
plt.plot(y_predictions, c='g', label='y_predictions')
plt.legend(loc='upper left')
plt.xlabel('Data points')
plt.ylabel('Energy (kWh)')
plt.title("Decision Tree Regression")
plt.ylim((-3000, 18000))
plt.xlim((0,150))
plt.savefig("treePLot")
plt.close()

#Random Forest model
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(x_train_set, y_train_set)
y_predictions = forest_reg.predict(x_test_set)

#Random Forest scores
forest_mse = mean_squared_error(y_test_set, y_predictions)
forest_rmse = np.sqrt(forest_mse)
print "Mean Squared Error: %s" %forest_rmse
forest_mae = mean_absolute_error(y_test_set, y_predictions)
print "Mean Absolute Error: %s" %forest_mae

#plot results
plt.plot(y_test_set, c='b', label='y_test_set')
plt.plot(y_predictions, c='g', label='y_predictions')
plt.legend(loc='upper left')
plt.xlabel('Data points')
plt.ylabel('Energy (kWh)')
plt.title("Random Forest Regression")
plt.ylim((-3000, 18000))
plt.xlim((0,150))
plt.savefig("forestPLot")
plt.close()

#Support Vector Machine model
svm_reg = SVR(kernel="linear")
svm_reg.fit(x_train_set, y_train_set)
y_predictions = svm_reg.predict(x_test_set)

#Support Vector Machine scores
svm_mse = mean_squared_error(y_test_set, y_predictions)
svm_rmse = np.sqrt(svm_mse)
print "Mean Squared Error: %s" %svm_rmse
svm_mae = mean_absolute_error(y_test_set, y_predictions)
print "Mean Absolute Error: %s" %svm_mae

#plot results
plt.plot(y_test_set, c='b', label='y_test_set')
plt.plot(y_predictions, c='g', label='y_predictions')
plt.legend(loc='upper left')
plt.xlabel('Data points')
plt.ylabel('Energy (kWh)')
plt.title("Support Vector Machine Regression")
plt.ylim((-3000, 18000))
plt.xlim((0,150))
plt.savefig("svmPLot")
plt.close()

# PolynomialFeatures (prepreprocessing) 
poly = PolynomialFeatures(degree=3)
X_train = poly.fit_transform(x_train_set)
X_test = poly.fit_transform(x_test_set)

#linear regression model
linpoly = LinearRegression()
linpoly.fit(x_train_set, y_train_set)
y_predictions = linpoly.predict(x_test_set)

#linear regression scores
linpoly_mse = mean_squared_error(y_test_set, y_predictions)
linpoly_rmse = np.sqrt(linpoly_mse)
print "Mean Squared Error: %s" %linpoly_rmse
linpoly_mae = mean_absolute_error(y_test_set, y_predictions)
print "Mean Absolute Error: %s" %linpoly_mae

#plot results
plt.plot(y_test_set, c='b', label='y_test_set')
plt.plot(y_predictions, c='g', label='y_predictions')
plt.legend(loc='upper left')
plt.xlabel('Data points')
plt.ylabel('Energy (kWh)')
plt.title("Linear Regression with poly features")
plt.ylim((-3000, 18000))
plt.xlim((0,150))
plt.savefig("linpolyPLot")
plt.close()

scores = cross_val_score(forest_reg, x_test_set, y_test_set,
                         scoring="neg_mean_squared_error", cv=10)
#scores = cross_val_score(forest_reg, x_test_set, y_test_set, cv=10)
forest_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(forest_rmse_scores)
