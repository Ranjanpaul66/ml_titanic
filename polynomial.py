import pandas as pd
import operations
from sklearn import linear_model,preprocessing
from sklearn.metrics import make_scorer

train = pd.read_csv('train.csv')
operations.pre_proc_check_data(train)

target = train['Survived'].values
features = train[['Pclass', 'Age', 'Sex', 'SibSp','Parch']].values

classifier = linear_model.LogisticRegression()
classifier_fit = classifier.fit(features,target)

print(classifier_fit.score(features,target))
# print(make_scorer(features,target))
poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

poly_features_fit = classifier.fit(poly_features,target)
print(classifier_fit.score(poly_features,target))
# print(make_scorer(poly_features_fit,target))
