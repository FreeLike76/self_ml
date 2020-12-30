from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X, y = load_digits(n_class=2, return_X_y=True)
plt.matshow(X[0].reshape(8, 8), cmap=plt.gray())
#hiding marks
#plt.xticks(())
#plt.yticks(())
plt.show()
