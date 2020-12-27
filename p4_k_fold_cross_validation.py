import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df["isMale"] = df["Sex"] == "male"
X = df[["Age", "Fare", "Pclass", "isMale", "Siblings/Spouses", "Parents/Children"]].values
y = df["Survived"].values

resScore = 0
splits = 10

kf = KFold(n_splits=splits, shuffle=True)
for train, test in kf.split(X):
    #cur split
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    #model creation
    model = LogisticRegression()
    model.fit(X_train, y_train)
    #res
    print("Score:", model.score(X_test, y_test))
    resScore += model.score(X_test, y_test)

print("-"*splits)
print("Avg score: ", resScore/splits)
