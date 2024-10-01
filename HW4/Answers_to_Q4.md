# Part a
The majority of days are "Up" days.
All lags and the feature Today are all normally distributed. All the lags are identically distributed.
Volume has a heavy right tail.

# Confusion Matrices, Accuracies
<img width="259" alt="image" src="https://github.com/user-attachments/assets/17781c7e-3f2b-436c-ba16-b5584a0ee3c4">



# Part g
The method that performs the best depends on the metric we want to maximize/minimize.
While logistic regression has the highest accuracy, it has the highest number of false positives.
To predict the market direction, we want to minimize the number of false positives if we are looking to buy stocks.
If we are looking to sell stocks, we want to minimize the number of false negatives.
Naive Bayes has the lowest number of false positives, so if we want to lower the risk of buying stocks that will go down, we should use Naive Bayes.

This can be because Naive Bayes does well to generalize while not overreacting to noise in the data. Naive Bayes also does not require
as much data as other methods, so it can be more robust to overfitting.
