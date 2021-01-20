# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 23:42:54 2020

@author: guindo
"""

from sklearn.decomposition import FastICA

ica = FastICA(n_components=3,random_state=0)
X_ica = ica.fit_transform(X)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=3)

# run an LDA and use it to transform the features
X_lda = lda.fit(X, y).transform(X)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_lda.shape[1])



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=50)


# Fitting SVM to the dataset

from sklearn.svm import SVC
svm = SVC(kernel ='rbf',random_state=0,C=100,gamma= 0.001,)
svm.fit(X_train, y_train)

#prediction train and test
Y_predtraining = svm.predict(X_train)
Y_predtest = svm.predict(X_test)

#metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cmtrain = confusion_matrix( y_train, Y_predtraining)
cmtest = confusion_matrix( y_test, Y_predtest)

accuracy_train = accuracy_score( y_train, Y_predtraining)
accuracy_test = accuracy_score( y_test, Y_predtest)

#Cross validation example
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=svm, 
                             X=X,
                             y=y, 
                             cv=10,
                             n_jobs=1)
mean = accuracies.mean()
