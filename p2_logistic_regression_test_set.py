import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")

df["isMale"] = df["Sex"] == "male"

X = df[["Age", "Fare", "Pclass", "isMale", "Siblings/Spouses", "Parents/Children"]].values
y = df["Survived"].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)
#model results
y_pred = model.predict(X_test)
#precision model results
y_prec_pred = model.predict_proba(X_test)[:, 1] > 0.75

print("Accuracy:", accuracy_score(y_test, y_pred))              #0.7837837837837838
print("0.75 Accuracy:", accuracy_score(y_test, y_prec_pred))    #0.7702702702702703
print("Precision:", precision_score(y_test, y_pred))            #0.7272727272727273
print("0.75 Precision:", precision_score(y_test, y_prec_pred))  #0.9512195121951219
print("Recall:", recall_score(y_test, y_pred))                  #0.7272727272727273
print("0.75 Recall:", recall_score(y_test, y_prec_pred))        #0.4431818181818182
print("f1 score:", f1_score(y_test, y_pred))                    #0.7272727272727273
print("0.75 f1 score:", f1_score(y_test, y_prec_pred))          #0.6046511627906976
