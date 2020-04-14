import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from functions import estimate_gaussian_params, train_validation_splits, selectThreshold, metrics


df = pd.read_csv('creditcardfraud/creditcard.csv')
classes = df['Class']
df.drop(['Time', 'Class'], axis=1, inplace=True)
cols = df.columns.difference(['Class'])
MMscaller = MinMaxScaler()
df = MMscaller.fit_transform(df)
df = pd.DataFrame(data=df, columns=cols)
df = pd.concat([df, classes], axis=1)

# missing values
print("missing values:", df.isnull().values.any())

# plot normal and fraud
count_classes = pd.value_counts(df['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), ['Normal', 'Fraud'])
plt.xlabel("Class")
plt.ylabel("Frequency")
#plt.show()

# heatmap big
sns.heatmap(df.corr(), vmin=-1)
plt.show()

"""fig, axs = plt.subplots(6, 5, squeeze=False)
for i, ax in enumerate(axs.flatten()):
    ax.set_facecolor('xkcd:charcoal')
    ax.set_title(df.columns[i])
    sns.distplot(df.iloc[:, i], ax=ax, fit=norm,
                color="#DC143C", fit_kws={"color": "#4e8ef5"})
    ax.set_xlabel('')
fig.tight_layout(h_pad=-1.5, w_pad=-1.5)
plt.show()"""

(Xtrain, Xtest, Xval, Ytest, Yval) = train_validation_splits(df)

(mu, sigma) = estimate_gaussian_params(Xtrain)

# calculate gaussian pdf
p = multivariate_normal.pdf(Xtrain, mu, sigma)
pval = multivariate_normal.pdf(Xval, mu, sigma)
ptest = multivariate_normal.pdf(Xtest, mu, sigma)

(epsilon, F1) = selectThreshold(Yval, pval)

print("Best epsilon found:", epsilon)
print("Best F1 on cross validation set:", F1)
print("Outliers found:", np.sum(p < epsilon))

(test_precision, test_recall, test_F1) = metrics(Ytest, ptest < epsilon)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 score:", test_F1)
