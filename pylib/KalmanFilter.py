

"""
	Kalman filter.
	https://en.wikipedia.org/wiki/Kalman_filter#Details

	Version 1 2019-09-05 Abhinav Kumar
"""	

import os,sys
import numpy as np
from numpy.linalg import inv

def predict(A, Q, s, P):
	s_pred = A.dot(s)
	P_pred = np.matmul(np.matmul(A, P), A.transpose())

	return s_pred, P_pred

def update(H, R, z, s_pred, P_pred):
	# K = P_pred H^T (H P_pred H^T + R)^{-1}
	temp2 = np.matmul(np.matmul(H, P_pred), H.transpose())
	temp = np.linalg.inv(temp2 + R)
	K = np.matmul(P_pred, np.matmul(H.transpose(), temp))
	
	# wt = I- K H
	I = np.eye(K.shape[0])
	wt = (I- np.matmul(K, H))

	# s
	s =  K.dot(z) + wt.dot(s_pred)
	P = wt.dot(P_pred)

	return s, P

def kalman_filter(A, Q, H, R, z, s, P):
	"""
		s_t = A s_{t-1} + w_t    w_t ~ N(0, Q)
		z_t = H s_t     + v_t    v_t ~ N(0, R)
	"""
	s_pred, P_pred = predict(A, Q, s, P)
	s_new, P_new   = update(H, R, z, s_pred, P_pred)

	return s_new, P_new