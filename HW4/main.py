import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


print("Part a")

stock_data = pd.read_csv("WeeklyStockData.csv")

stock_data.head()

numerical_summary = stock_data.describe()

direction_distribution = stock_data['Direction'].value_counts()

print(numerical_summary, direction_distribution)

# The majority of days are "Up" days.
# All lags and the feature Today are all normally distributed. All the lags are identically distributed.
# Volume has a heavy right tail.


columns = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Today', 'Volume']

for column in columns:
	stock_data[column].hist(bins=30)
	plt.title(f'Histogram of {column}')
	plt.xlabel(column)
	plt.ylabel('Frequency')
	plt.show()

plt.figure(figsize=(6, 4))
stock_data['Direction'].value_counts().plot(kind='bar')
plt.title('Direction Distribution')
plt.xlabel('Direction')
plt.ylabel('Frequency')
plt.show()

print("Part b")

train_data = stock_data[stock_data['Year'] <= 2008]
test_data = stock_data[stock_data['Year'] > 2008]

X_train = train_data[['Lag2']]
y_train = train_data['Direction'].map({'Up': 1, 'Down': 0})

X_test = test_data[['Lag2']]
y_test = test_data['Direction'].map({'Up': 1, 'Down': 0})

model = LogisticRegression(solver='liblinear')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(conf_matrix, accuracy)

print("Part c")


X_train_lda = train_data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
X_test_lda = test_data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]

model_LDA = LinearDiscriminantAnalysis(solver='lsqr')
model_LDA.fit(X_train_lda, y_train)

y_pred_lda = model_LDA.predict(X_test_lda)

conf_matrix_lda = confusion_matrix(y_test, y_pred_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)

print(conf_matrix_lda, accuracy_lda)

print("Part d")

modelQDA = QuadraticDiscriminantAnalysis()
modelQDA.fit(X_train_lda, y_train)

y_pred_qda = modelQDA.predict(X_test_lda)

conf_matrix_qda = confusion_matrix(y_test, y_pred_qda)
accuracy_qda = accuracy_score(y_test, y_pred_qda)

print(conf_matrix_qda, accuracy_qda)

print("Part e")


modelKNN = KNeighborsClassifier(n_neighbors=1)
modelKNN.fit(X_train_lda, y_train)

y_pred_knn = modelKNN.predict(X_test_lda)

conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(conf_matrix_knn, accuracy_knn)


print("Part f")
modelNaiveBayes = GaussianNB()
modelNaiveBayes.fit(X_train_lda, y_train)

y_pred_nb = modelNaiveBayes.predict(X_test_lda)

conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(conf_matrix_nb, accuracy_nb)

print("Part g")
'''
The method that performs the best depends on the metric we want to maximize/minimize.
While logistic regression has the highest accuracy, it has the highest number of false positives.
To predict the market direction, we want to minimize the number of false positives if we are looking to buy stocks.
If we are looking to sell stocks, we want to minimize the number of false negatives.
Naive Bayes has the lowest number of false positives, so if we want to lower the risk of buying stocks that will go down, we should use Naive Bayes.

This can be because Naive Bayes does well to generalize while not overreacting to noise in the data. Naive Bayes also does not require
as much data as other methods, so it can be more robust to overfitting.
'''
