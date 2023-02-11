import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np


# data set at : https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression?resource=download

# opening data file in such a way that the file is closed after reading data
# this is better for memory consumption. Noticeable with larger data files
with open("C:/Users/User/Desktop/Real estate.csv") as file:
    data = pd.read_csv(file)

data.columns = ['index', 'date', 'house_age', 'mrt_distance', 'stores_nearby', 'x5', 'x6', 'price_in_millions']
# file has too many columns which arent too useful
# dropping some columns
data = data.drop(columns=['index', 'x5', 'x6'])

print(data.head(10))  # get some idea of the data
print("________________________________________")
print(data.describe())  # get some idea of the data
print("________________________________________")
print(data.info())  # this allows us to check if there are any null values.
# since no null values we can ignore data cleaning to replace any null values or correct for it.


# since we have many variables, it is useful to observe the correlation
correlation_data = data.iloc[:, 1:5]  # for correlation, we only pass in the numerical input values
C = correlation_data.corr()

sns.heatmap(data=C, mask=np.zeros_like(C), cmap=sns.diverging_palette(220, 10, as_cmap=True), linewidths=0.5)
plt.show()


#show a scatter plot to understand how the relationship in the data is.
sns.scatterplot(data = data, x = "mrt_distance", y = "price_in_millions" )
plt.show()

sns.scatterplot(data = data, x = "house_age", y = "price_in_millions" )
plt.show()


sns.scatterplot(data = data, x = "stores_nearby", y = "price_in_millions" )
plt.show()

sns.pairplot(data =data) # using this with a data set that is applicable for logisitcs is very insightful when the hue is defined by a column
plt.show()

# create an array of X and Y values
# These values are passed into the PCA for being computed
X = data.iloc[:, 0:3].values
Y = data.iloc[:, 4].values

#__________________________________________________________
#UNCOMMENT BELOW TO RUN PCA
# pca = decomposition.PCA(
#     n_components=2)  # decides how many components to keep, that is variables to be reduced from original to new count of variables
# pca.fit(X)
# X = pca.transform(X)
# print(X)  # shows how the fields are changed from the original to get an idea of the PCA that was done

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3)  # test and train split with 70-30 break down

lr = LinearRegression()  # create an object of the model
lr.fit(X_train, Y_train)  # give the training data for fitting the model
print(lr.score(X_test,
               Y_test))  # gives the mean accuracy of the model after the initial fitting is done using the test data


Y_predicted = lr.predict(X_test) # predict values for X_Test

#comparing the mrt distance to the house price
plt.scatter(X_test[:,1], Y_test, color ='b')
plt.scatter(X_test[:,1], Y_predicted, color ='k')
plt.show()

#comparing the house age to the house price
plt.scatter(X_test[:,0], Y_test, color ='b')
plt.scatter(X_test[:,0], Y_predicted, color ='k')
plt.show()

#comparing the stores nearby to the house price
plt.scatter(X_test[:,2], Y_test, color ='b')
plt.scatter(X_test[:,2], Y_predicted, color ='k')
plt.show()

