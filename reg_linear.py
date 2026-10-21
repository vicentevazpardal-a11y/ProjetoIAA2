import numpy as np 

def regressao_linear (X, Y):
    X = np.array(X)
    Y = np.array(Y)
    
    X_novo = np.column_stack((np.ones(X.shape[0]), X))
    
    theta = np.linalg.solve(X_novo.T @ X_novo, X_novo.T @ Y)
    
    beta = theta[1]
    beta_0 = theta[1:]
    
    return beta, beta_0
    