from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

#creating custom dataset with 2 features
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y)

#creating a model
#   -set a custom number of max_iter to 1000 because 200 was not enough
#   -set a custom params for hidden_layer to 100, 50
#in result accuracy increased
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50))
mlp.fit(X_train, y_train)

#scoring a model
print(mlp.score(X_test, y_test))