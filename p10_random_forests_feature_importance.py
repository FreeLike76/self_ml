import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data["data"], columns=cancer_data["feature_names"])
df["target"] = cancer_data["target"]

X = df[cancer_data["feature_names"]].values
y = df["target"].values

rf = RandomForestClassifier(random_state=100)
rf.fit(X, y)

rf_imp = pd.Series(rf.feature_importances_, index=cancer_data["feature_names"]).sort_values(ascending=False)

print(rf_imp.head())    #top important features
