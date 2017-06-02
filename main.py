import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from scipy.stats import randint as sp_randint

#PCA
from sklearn.decomposition import PCA
from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt

#For Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



df = pd.read_csv("breast_cancer_dataset.csv",header = 0)
print (df.head())

df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)

le = preprocessing.LabelEncoder()
le.fit(['M','B'])

df['diagnosis'] = le.transform(df['diagnosis'])
print (df.describe())
print (df.head())

#PCA
observables = df.iloc[:,1:]
pca = PCA(n_components=3)
pca.fit(observables)
dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
components = pd.DataFrame(np.round(pca.components_, 4), columns = observables.keys())
components.index = dimensions
ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
variance_ratios.index = dimensions

print(pd.concat([variance_ratios, components], axis = 1))


#Feature selection
tst = df.corr()['diagnosis'].copy()
tst = tst.drop('diagnosis')
tst.sort_values(inplace=True)
tst.plot(kind='bar', alpha=0.6)
plt.show()

malignant = df[df['diagnosis'] ==1]
benign = df[df['diagnosis'] ==0]

observe = list(df.columns[1:11]) + ['area_worst'] + ['perimeter_worst']
observables = df.loc[:,observe]


#Histograms
plt.rcParams.update({'font.size': 8})
plot, graphs = plt.subplots(nrows=6, ncols=2, figsize=(8, 10))
graphs = graphs.flatten()
for idx, graph in enumerate(graphs):
    graph.figure
    binwidth = (max(df[observe[idx]]) - min(df[observe[idx]])) / 50
    bins = np.arange(min(df[observe[idx]]), max(df[observe[idx]]) + binwidth, binwidth)
    graph.hist([malignant[observe[idx]], benign[observe[idx]]], bins=bins, alpha=0.6, normed=True,
               label=['Malignant', 'Benign'], color=['red', 'blue'])
    graph.legend(loc='upper right')
    graph.set_title(observe[idx])
plt.tight_layout()
plt.show()



#Scatter_matrix
color_wheel = {0: "blue", 1: "red"}
colors = df["diagnosis"].map(lambda x: color_wheel.get(x))
scatter_matrix(observables, c=colors, alpha = 0.5, figsize = (15, 15), diagonal = 'kde');
plt.show()


observables.drop(['fractal_dimension_mean', 'smoothness_mean', 'symmetry_mean'],axis=1,inplace=True)


# Classification
X = observables
y = df['diagnosis']


# -------------------Naive Bayes
gnb = GaussianNB()
gnb_scores = cross_val_score(gnb, X, y, cv=10, scoring='accuracy')
print(gnb_scores.mean())
#To know how these scores are predicted refer your notebook 8/5/2017

gnb.fit(X,y)

# This I used to display the row of the selected columns
print (observables.ix[1])


#gnb.fit(X,y)
#print(gnb.predict([13.54,14.36,87.46,566.3,0.08129,0.06664,0.04781,711.2,99.7]))



'''
#------------------------KNN
knn = KNeighborsClassifier()

k_range = list(range(1, 30))
leaf_size = list(range(1,30))
weight_options = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options, 'algorithm': algorithm}

#Randomizedsearch is a tuning algorithm that accepts which algorithm to use (knn) and the
# parameter(param_grid) to pass with the algorith and then the parameter for itself i.e the tuning algo
rand_knn = RandomizedSearchCV(knn, param_grid, cv=10, scoring="accuracy", n_iter=100, random_state=42)
rand_knn.fit(X,y)

print(rand_knn.best_score_)
print(rand_knn.best_params_)
print(rand_knn.best_estimator_)
#print(rand_knn.predict([17.99,10.38,122.8,1001,0.2776,0.3001,0.1471,2019,184.6]))

'''
'''
#--------------------------Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'min_samples_split': sp_randint(2, 11),
              'min_samples_leaf': sp_randint(1, 11)}
rand_dt = RandomizedSearchCV(dt_clf, param_grid, cv=10, scoring="accuracy", n_iter=100, random_state=42)
rand_dt.fit(X,y)
print(rand_dt.best_score_)
print(rand_dt.best_params_)
print(rand_dt.best_estimator_)

#---------------------------Support Vector Machines
sv_clf = SVC(random_state=42)

param_grid = [
              {'C': [1, 10, 100, 1000],
               'kernel': ['linear']
              },
              {'C': [1, 10, 100, 1000],
               'gamma': [0.001, 0.0001],
               'kernel': ['rbf']
              },
 ]
grid_sv = GridSearchCV(sv_clf, param_grid, cv=10, scoring="accuracy")
grid_sv.fit(X,y)
print(grid_sv.best_score_)
print(grid_sv.best_params_)
print(grid_sv.best_estimator_)

#--------------------------------Random Forest
rf_clf = RandomForestClassifier(random_state=42)

param_grid = {"max_depth": [3, None],
              "max_features":  sp_randint(1, 8),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
rand_rf = RandomizedSearchCV(rf_clf, param_distributions=param_grid, n_iter=100, random_state=42)
rand_rf.fit(X,y)
print(rand_rf.best_score_)
print(rand_rf.best_params_)
print(rand_rf.best_estimator_)
'''

'''
#--------------------------------------Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(9,9,1),max_iter=1410)
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
grid_nn = GridSearchCV(estimator=mlp, param_grid=dict(alpha=alphas))
grid_nn.fit(X, y)
print(grid_nn.best_score_)
'''



'''
import gui as g
def predictor(x):
    print(eval(x).predict([float(g.entry1.get()),float(g.entry2.get()),float(g.entry3.get()),float(g.entry4.get()),float(g.entry5.get()),float(g.entry6.get()),float(g.entry7.get()),float(g.entry8.get()),float(g.entry9.get())]))
'''

def predictor(x,e1,e2,e3,e4,e5,e6,e7,e8,e9):
    print(eval(x).predict([e1,e2,e3,e4,e5,e6,e7,e8,e9]))
