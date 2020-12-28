import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from IPython.display import Image

df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
df["isMale"] = df["Sex"] == "male"

features = ["Age", "Fare", "Pclass", "isMale", "Siblings/Spouses", "Parents/Children"]

X = df[features].values
y = df["Survived"].values

dt = DecisionTreeClassifier()
dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=features)
graph = graphviz.Source(dot_file)
graph.render(filename="titanicTree", format="png", cleanup="True")
