import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data["data"], columns=cancer_data["feature_names"])
df["target"] = cancer_data["target"]

X = df[cancer_data["feature_names"]].values
y = df["target"].values

param_list = list(range(1, 101))
param_grid = {"n_estimators": param_list}

rf = RandomForestClassifier(random_state=100)
gs = GridSearchCV(rf, param_grid, scoring="f1", cv=5)
gs.fit(X, y)
scores = gs.cv_results_["mean_test_score"]

plt.plot(param_list, scores)
plt.xlabel("n_estimators")
plt.ylabel("accuracy (ok for balanced classes)")
plt.xlim(0, 100)
plt.ylim(0.9, 1)
plt.show()  #levels out at 10, best at 24 and 66
