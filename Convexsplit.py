import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator


# helper function for graph Laplacian
# returns normalized graph Laplacian
def _graph_laplacian(input_vector, tau):
    
    # This computes the graph laplacian according to equation 2.5
    
    data_size = input_vector.shape[0]

    pairwise_sq_dists = squareform(pdist(input_vector, 'sqeuclidean'))
    L = -scipy.exp(-pairwise_sq_dists / tau) + np.eye(data_size)

    for k in range(data_size):
        L[k, k] = -np.sum(L[k, :])

    # compute inverse square root of diagonal matrix
    d = np.diag(L)
    D = np.mat(np.diag(d))
    D_inv_sqrt = np.zeros(D.shape)
    np.mat(np.fill_diagonal(D_inv_sqrt, 1 / (np.array(D.diagonal())[0] ** 0.5)))

    # compute normalized graph laplacian by equation 2.7
    Ls = D_inv_sqrt.dot(L).dot(D_inv_sqrt)

    return Ls


# helper function for convexsplit
def _convexsplit(eigenvectors, eigenvalues, u0, train_size, c=1, epsilon=2, maxit=50):
    
    #This method performs the convexsplit scheme to minimize energy potential

    # eigenvectors is a matrix whose columns are eigenvectors Phi_i
    # eigenvalues is an array whose entries are corresponding eigenvalues Lambda_i
    # here we deal with the case when eigenvectors is a square matrix

    dt = 0.1
    k = eigenvalues.size
    n = u0.shape[0]

    # initialize matrices for storage
    # this part can be optimized
    # since we do not need to store the result from each iteration
    A = np.zeros([k, maxit + 1])
    B = np.zeros([k, maxit + 1])
    D = np.zeros([k, maxit + 1])
    F = np.zeros(k)
    U = np.zeros([n, maxit + 1])

    # initialize u_0 and lambda
    # gamma here denotes lambda in the original formula
    U[:, 0] = np.zeros(n)
    U[0: train_size, 0] = u0[0: train_size].transpose()
    gamma = np.zeros(n)
    gamma[0: train_size] = np.ones(train_size)

    # decompose u0 u0.^3 as sums of Phi_i
    # where 0 <= i <= k-1
    a1 = np.linalg.solve(eigenvectors, U[:, 0])
    u0_cube = np.array(U[:, 0]) ** 3
    b1 = np.linalg.solve(eigenvectors, u0_cube)
    A[:, 0] = a1
    B[:, 0] = b1
    F = 1 + dt * (epsilon * eigenvalues + c)

    # convexsplit iteration
    for j in range(maxit):
        A[:, j + 1] = (F ** -1) * ((1 + dt / epsilon + c * dt) \
                        * A[:, j] - dt / epsilon * B[:, j] - dt * D[:, j])
        U[:, j + 1] = eigenvectors.dot(A[:, j + 1])
        B[:, j + 1] = np.linalg.solve(eigenvectors, U[:, j + 1] ** 3)
        D[:, j + 1] = np.linalg.solve(eigenvectors, 
                        np.multiply(gamma, U[:, j + 1] - U[:, 1]))

    return U[:, maxit]


class ConvexsplitClassifier(BaseEstimator, ClassifierMixin):
    # This is the sklearn style interface
    
    def __init__(self, tau=0.3, c=1, epsilon=2, maxit=500):
        # Initialize classifier
        
        self.tau = tau
        self.c = c
        self.epsilon = epsilon
        self.maxit = maxit
        self.x_train = None
        self.y_train = None


    def fit(self, x_train, y_train):
        # For a standard fit method,
        # you should verify all inputs are valid here

        self.x_train = x_train
        self.y_train = np.copy(y_train)
        self.y_train[self.y_train == 0] = -1 # negative cases are labelled as -1


    def predict(self, x_test):


        train_size = self.x_train.shape[0]

        # combine train set with test set
        X = np.concatenate((self.x_train, x_test), axis=0)
        
        # compute laplacian of combined data
        L = _graph_laplacian(X, tau=self.tau)
        
        # compute eigenvalues and eigenvectors of laplacian
        Lam, Phi = np.linalg.eig(L)
        
        # initialize u0
        u0 = np.zeros(X.shape[0])
        u0[0:train_size] = self.y_train

        y_pred = _convexsplit(Phi, Lam, u0, train_size)[train_size:]

        # sign(u) gives the predicted class
        # format the output to 0 and 1
        y_pred[y_pred <= 0] = 0
        y_pred[y_pred > 0] = 1

        return y_pred