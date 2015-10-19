__author__ = 'ragav777'
import numpy as np
import scipy.optimize as op

# CostFunction gets a X that is  m x (n+1) for theta0
# y is m x 1 theta is (n+1) x 1
def costfunction(theta, X, y, lda ):
    m,n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    term1 = (1/(2*m)) * sum(np.square(error))
    temp2 = np.vstack((np.zeros((1,1), dtype = np.int), theta[1:n]))
    term2 = (lda/(2*m))*sum(np.square(temp2))
    J = term1 + term2
    return J

def gradient(theta, X, y, lda ):
    m,n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    term1 = (1/m) * np.dot((X.T), error)
    temp2 = np.vstack((np.zeros((1,1), dtype = np.int), theta[1:n]))
    term2 = (lda/m)*temp2
    grad = term1 + term2
    return grad.flatten()

def trainlinearregression ( X, y, lda):
    m,n = X.shape
    print (str(m) + " " + str(n) )
    initial_theta = np.zeros((n, 1))
    result = op.minimize(fun = costfunction, x0 = initial_theta, args = (X, y, lda), method = 'TNC',
             jac = gradient, options ={ 'disp': False, 'maxiter': 200 }  )
    optimal_theta = result.x
    return optimal_theta

def main():
    X = np.array([[1,1,2,3],[1,4,5,6], [1,7,8,9]])
    y = np.array([21,48,75]).reshape((3,1))
    # theta = np.array([1,2,3,4])
    lda = 0.001
    # J = costfunction(X, y, theta, lda)
    # G = gradient(X, y, theta, lda)
    t = trainlinearregression( X, y, lda)
    print(t)
    # print(G)


if __name__ == '__main__' :
    main()
else :
    print ("Didn't Work")