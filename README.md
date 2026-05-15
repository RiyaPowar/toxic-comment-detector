# toxic-comment-detector

TF-IDF is used for feature extraction and Multi-Output Classifier is used for multi-label classification.
LogisticRegression with max_iter=1000 and class_weight='balanced' is used for classification.
TF-IDF gives numerical representation of data based on word frequency and inverse document frequency.
max_iter=1000 is used to increase the number of iterations for convergence.
class_weight='balanced' is used to handle the imbalance in the dataset.

Uses optimization algorithms (called solvers)
These solvers internally handle step sizes / learning rates

solver = 'lbfgs'
L-BFGS is a quasi-Newton method
It does not use a fixed learning rate like gradient descent
Instead, it:
Approximates second-order derivatives (curvature)
Automatically determines step size per iteration
Good for small and medium datasets


Joblib is used for efficient serialization of Python objects, especially:
NumPy arrays
Large ML models

Compared to pickle, it’s:
Faster
More memory-efficient for big data
