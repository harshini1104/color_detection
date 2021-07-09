from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris = datasets.load_iris()
type(iris)

print(iris.keys())
type(iris.data), type(iris.target)

#no of rows and columns respectively
iris.data.shape

iris.target_names

#EDA
X = iris.data
Y = iris.target
df = pd.DataFrame(X, columns = iris.feature_names)
print(df.head())

pd.plotting.scatter_matrix(df, c=Y, figsize = [8,8], s=150, marker = 'D')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])

iris['data'].shape, iris['target'].shape # there are 150 labels

X_new = np.array([[5.6, 2.8, 3.9, 1.1], [5.7, 2.6, 3.8, 1.3], [4.7, 3.2, 1.3, 0.2]])
prediction = knn.predict(X_new)
X_new.shape

print('Prediction: {}'.format(prediction))
