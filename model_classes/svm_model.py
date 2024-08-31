import numpy as np
from cvxopt import matrix, solvers

class SVMAudioClassifier:
    def __init__(self, C=1.0, kernel='linear', gamma=None, coef0=0.0):
        self.C = C                              # Penalty parameter or regularization term
        self.kernel = kernel                    # Kernel type: 'linear' or 'rbf'
        self.gamma = gamma                      # Kernel coefficient for 'rbf' kernel
        self.coef0 = coef0                      # Independent term in kernel function (not used here)
        self.lagr_multipliers = None            # Lagrange multipliers obtained from the optimization problem
        self.support_vectors = None             # Support vectors from the training data
        self.support_vector_labels = None       # Support vectors from the training labels
        self.intercept = None                   # Intercept term for the decision boundary
        
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        
        # Set default gamma if not provided (1 / n_features)
        if not self.gamma:
            self.gamma = 1 / n_features
            
        # Select the kernel function based on the input parameter
        if self.kernel == 'linear':
            kernel = self.linear_kernel
        elif self.kernel == 'rbf':
            kernel = self.rbf_kernel
        else:
            raise ValueError("Invalid kernel type. Only 'linear' and 'rbf' are supported.")
        
        # Compute the Gram matrix (kernel matrix)
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = kernel(X[i], X[j])
                
        # Define the quadratic optimization problem using the kernel matrix
        P = matrix(np.outer(y, y) * K, tc='d')              # Matrix P for quadratic term
        q = matrix(-1 * np.ones(n_samples), tc='d')         # Vector q for linear term
        A = matrix(y, (1, n_samples), tc='d')               # Equality constraint: sum(alpha_i * y_i) = 0
        b = matrix(0, tc='d')                               # Equality constraint term
        
        # Hard margin case: no regularization
        if not self.C:
            G = matrix(np.diag(np.ones(n_samples) * -1), tc='d')
            h = matrix(np.zeros(n_samples), tc='d')

        # Soft margin case, with regularization
        else:
            G_max = np.diag(np.ones(n_samples) * -1)
            G_min = np.identity(n_samples)
            G = matrix(np.vstack((G_max, G_min)), tc='d')
            h_max = matrix(np.zeros(n_samples), tc='d')
            h_min = matrix(np.ones(n_samples) * self.C, tc='d')
            h = matrix(np.vstack((h_max, h_min)), tc='d')
            
        minimization = solvers.qp(P, q, G, h, A, b)         # Solve the quadratic optimization problem using cvxopt
        lagr_mult = np.ravel(minimization['x'])             # Extract Lagrange multipliers

        # Extract the support vectors (where Lagrange multipliers are non-zero)
        idx = lagr_mult > 1e-5
        self.lagr_multipliers = lagr_mult[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]
        
        self.intercept = self.support_vector_labels[0]      # Calculate the intercept (bias term)
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[i] * kernel(self.support_vectors[i], self.support_vectors[0])
            
    def predict(self, X):
        if self.kernel == 'linear':
            kernel = self.linear_kernel
        elif self.kernel == 'rbf':
            kernel = self.rbf_kernel
        else:
            raise ValueError("Invalid kernel type. Only 'linear' and 'rbf' are supported.")
        
        y_pred = []                                         # Initialize list to store predictions
        for sample in X:
            prediction = 0
            for i in range(len(self.lagr_multipliers)):     # Sum the contributions of each support vector
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[i] * kernel(self.support_vectors[i], sample)
            prediction += self.intercept                    # Add the intercept term
            y_pred.append(np.sign(prediction))              # Predict the sign of the decision function
        return np.array(y_pred)
    
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def rbf_kernel(self, x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-self.gamma * distance)
    
    def margin(self, X):
        margins = []
        for sample in X:
            margin = 0
            for i in range(len(self.lagr_multipliers)):
                margin += self.lagr_multipliers[i] * self.support_vector_labels[i] * self.rbf_kernel(self.support_vectors[i], sample)
            margin += self.intercept
            margins.append(margin)
        return np.array(margins)
