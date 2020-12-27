import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

#df creation
df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df["isMale"] = df["Sex"] == "male"
X = df[["Age", "Fare", "Pclass", "isMale", "Siblings/Spouses", "Parents/Children"]].values
y = df["Survived"].values
#train&test data split
X_train, X_test, y_train, y_test = train_test_split(X, y)
#model creation
model = LogisticRegression()
model.fit(X_train, y_train)
#train proba
y_pred_proba = model.predict_proba(X_test)
#roc curve
false_pos_rates, true_pos_rates, threshholds = roc_curve(y_test, y_pred_proba[:, 1])
#ploting
plt.plot(false_pos_rates, true_pos_rates, c="green")
plt.plot([0, 1], [0, 1], c="yellow")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("1- specificity")
plt.ylabel("sensitivity")
plt.show()
#upper-left corner = good, below diagonal line = worse then random
#sensitovity = catch as many pos cases as possible
#specificity = making sure that founded pos cases are really TRUE is more important that finding all pos cases