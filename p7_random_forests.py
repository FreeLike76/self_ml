import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data["data"], columns=cancer_data["feature_names"])
df["target"] = cancer_data["target"]

X = df[cancer_data["feature_names"]].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)

print("Random forest score:\t", rf.score(X_test, y_test))
print("Logistic Regression score:\t", lr.score(X_test, y_test))
