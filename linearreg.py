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
             jac = gradient, options ={ 'disp': True, 'maxiter': 200 }  )
    optimal_theta = result.x
    return optimal_theta

def cost(theta, X, y):
    m,n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    error = np.dot(X, theta) - y
    sqe = sum(np.square(error))
    return error,sqe


def main():

    lda = 0.01
    Xtemp = np.loadtxt('house_train.csv', dtype = float, delimiter = ',', usecols = range(11) )
    mtr,ntr = np.shape(Xtemp)
    Xtrain = np.hstack ((np.ones ((mtr, 1)), Xtemp))
    Ytrain = np.loadtxt('house_train.csv', dtype = float, delimiter = ',', usecols = (11,) )

    Xtemp = np.loadtxt('house_cv.csv', dtype = float, delimiter = ',', usecols = range(11) )
    mcv,ncv = np.shape(Xtemp)
    Xcv = np.hstack ((np.ones ((mcv, 1)), Xtemp))
    Ycv = np.loadtxt('house_cv.csv', dtype = float, delimiter = ',', usecols = (11,) )

    Xtemp = np.loadtxt('house_test.csv', dtype = float, delimiter = ',', usecols = range(11) )
    mtst,ntst = np.shape(Xtemp)
    Xtest = np.hstack ((np.ones ((mtst, 1)), Xtemp))
    Ytest = np.loadtxt('house_test.csv', dtype = float, delimiter = ',', usecols = (11,) )

    # print(Xtrain)
    t = trainlinearregression( Xtrain, Ytrain, lda)
    print(t)
    errtr, sqetr = cost(t, Xtrain, Ytrain)
    #print(errtr)

    errcv, sqecv = cost(t, Xcv, Ycv)
    print(errcv)


if __name__ == '__main__' :
    main()
else :
    print ("Didn't Work")