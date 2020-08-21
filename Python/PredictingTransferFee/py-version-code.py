### (0) Import libraries ###

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection._split import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import validation_curve

### (1) Data Import & EDA ###

data = pd.read_csv("../input/top-250-football-transfers-from-2000-to-2018/top250-00-19.csv")
print(data.info())

data.describe()

print(data.loc[data['Age'] < 15])

### (2) Data Cleaning ###

# Fixing age 0 player Marzouq Al-Otaibi to 25 according to TransferMarkt
data.loc[data['Age'] == 0, 'Age'] = 25

# Dropping relatively irrelevant columns
for drop_columns in ['Name', 'Team_from', 'League_from', 'Team_to', 'League_to']:
    data = data.drop(drop_columns, axis='columns')

# Redefining Seasons column to beginning year of season
data.Season = data.Season.str.slice(start=0, stop=4).astype(int)

# Handling NaN values of Market_value
data.Market_value.fillna(data.Transfer_fee, inplace=True)

# Changing positions into separate binary columns for machine learning
positionsArray = data.Position.unique().astype(str)
data = pd.concat((data,pd.get_dummies(data['Position'])),axis=1)
data = data.drop('Position', axis='columns')

# Check data
data.info()
data.head()

### (3) Machine Learning: Random Forest Regressor ###

X = data
y = data['Transfer_fee']

# Pick test size of 0.5 to avoid overfitting + gives better r^2 and lower rmse than 0.4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=42)
forest = RandomForestRegressor(random_state = 1)

# Performance on Test Set
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

rmsd = np.sqrt(mean_squared_error(y_test, y_pred))      
r2_value = r2_score(y_test, y_pred) 

print(rmsd)
print(r2_value)

# Performance on Training Set
forest.fit(X_test, y_test)
y_pred = forest.predict(X_train)

rmsd = np.sqrt(mean_squared_error(y_train, y_pred))      
r2_value = r2_score(y_train, y_pred) 

print(rmsd)
print(r2_value)

### (4) Hyperparameter Tuning ###

### (4.1) Number of Estimators
num_est = np.arange(100, 1000, 50)
train_scoreNum, test_scoreNum = validation_curve(
                                RandomForestRegressor(),
                                X = X_train, y = y_train, 
                                param_name = 'n_estimators', 
                                param_range = num_est, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scoreNum, axis=1)
train_std = np.std(train_scoreNum, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scoreNum, axis=1)
test_std = np.std(test_scoreNum, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(num_est, train_mean, label="Training score", color="black")
plt.plot(num_est, test_mean, label="Cross-validation score", color="dimgrey")

plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

# 700 is best

### (4.2) Max Depth
m_depth = [5, 10, 15, 20, 25, 30]
train_scoreNum, test_scoreNum = validation_curve(
                                RandomForestRegressor(),
                                X = X_train, y = y_train, 
                                param_name = 'max_depth', 
                                param_range = m_depth, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scoreNum, axis=1)
train_std = np.std(train_scoreNum, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scoreNum, axis=1)
test_std = np.std(test_scoreNum, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(m_depth, train_mean, label="Training score", color="black")
plt.plot(m_depth, test_mean, label="Cross-validation score", color="dimgrey")

plt.title("Max_Depth")
plt.xlabel("Number Of Trees")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

# Pick 10

### (4.3) Minimum Samples Split
m_split = [2, 5, 10, 15, 20, 25, 30]
train_scoreNum, test_scoreNum = validation_curve(
                                RandomForestRegressor(),
                                X = X_train, y = y_train, 
                                param_name = 'min_samples_split', 
                                param_range = m_split, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scoreNum, axis=1)
train_std = np.std(train_scoreNum, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scoreNum, axis=1)
test_std = np.std(test_scoreNum, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(m_split, train_mean, label="Training score", color="black")
plt.plot(m_split, test_mean, label="Cross-validation score", color="dimgrey")

plt.title("Validation Curve With Random Forest")
plt.xlabel("Min_Samples_Split")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

# Pick 2

### (4.4) Minimum Samples Leaf
m_leaf = [1, 2, 4, 6, 8, 10]
train_scoreNum, test_scoreNum = validation_curve(
                                RandomForestRegressor(),
                                X = X_train, y = y_train, 
                                param_name = 'min_samples_leaf', 
                                param_range = m_leaf, cv = 3)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scoreNum, axis=1)
train_std = np.std(train_scoreNum, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scoreNum, axis=1)
test_std = np.std(test_scoreNum, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(m_leaf, train_mean, label="Training score", color="black")
plt.plot(m_leaf, test_mean, label="Cross-validation score", color="dimgrey")

plt.title("Validation Curve With Random Forest")
plt.xlabel("Min_Samples_Leaf")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

# Pick 2 

# (5) Machine Learning With Hyperparameter Tuning: Random Forest Regressor

X = data
y = data['Transfer_fee']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=42)
forest = RandomForestRegressor(random_state = 1, n_estimators = 300, max_depth = 15, min_samples_split= 2, min_samples_leaf = 2)

# Performance on Test Set
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

rmsd = np.sqrt(mean_squared_error(y_test, y_pred))      
r2_value = r2_score(y_test, y_pred) 

print(rmsd)
print(r2_value)

# Performance on Training Set
forest.fit(X_test, y_test)
y_pred = forest.predict(X_train)

rmsd = np.sqrt(mean_squared_error(y_train, y_pred))      
r2_value = r2_score(y_train, y_pred) 

print(rmsd)
print(r2_value)

# Pick pre-tuned model because performance on training set increases after tuning but on test set it decreases, a hint of overfitting
