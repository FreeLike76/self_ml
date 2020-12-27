import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#df = pd.read_csv("resources/titanic.csv")
df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")

X = df[["Age", "Fare"]].values
Y = df["Survived"].values

model = LogisticRegression()
model.fit(X, Y)
print(model.score(X, Y))

lineX = np.linspace(0, 100, 100)
lineY = -(model.coef_[0, 0]*lineX + model.intercept_)/model.coef_[0, 1]

plt.scatter(df["Age"], df["Fare"], c=df["Survived"])
plt.plot(lineX, lineY, c="red")

plt.title("0=(" + str(model.coef_[0, 0]) + ")*X+(" + str(model.coef_[0, 1]) + ")*Y+(" + str(model.intercept_[0]) + ")")

plt.xlabel("Age")
plt.ylabel("Fare")

plt.grid()
plt.show()

