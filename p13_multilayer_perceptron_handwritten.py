from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

print(mlp.score(X_test, y_test))
#1 for binary
#~0.97 for 0-9

#looking at wrong predictions
y_pred = mlp.predict(X_test)
X_incorrect = X_test[y_pred != y_test]
y_incorrect_real = y_test[y_pred != y_test]
y_incorrect_pred = y_pred[y_pred != y_test]

for i in range(0, len(y_incorrect_pred)):
    print("predicted:", y_incorrect_pred[i], "real:", y_incorrect_real[i])