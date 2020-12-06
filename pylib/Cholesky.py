

import numpy as np

def cholesky_to_covar(L_vect, laplacian= False):
    """
        Converts cholesky numpy array to covariance matrix and also gets the 
        determinant of covariance
    """
    L_mat = np.zeros((L_vect.shape[0], L_vect.shape[1], 2, 2))
    # Reshape L_0, L_1, L_2 into 2x2 Lower triangular matrix
    # [ L_0   0  ]
    # [ L_1   L_2]
    L_mat[:, :, 0, 0] = L_vect[:, :, 0]
    L_mat[:, :, 1, 0] = L_vect[:, :, 1]
    L_mat[:, :, 1, 1] = L_vect[:, :, 2]

    sigma = np.matmul(L_mat, np.transpose(L_mat, (0, 1, 3, 2)))
    sigma = scale_covariance(sigma, laplacian= laplacian)

    # Get determinant of the covariance
    det_cov = np.zeros((sigma.shape[0],sigma.shape[1]))
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            det_cov[i,j] = np.linalg.det(sigma[i,j,:,:])

    return sigma, det_cov
    
    
def scale_covariance(covar, laplacian= False):
    
    return covar
