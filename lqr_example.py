import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.linalg import inv

A = np.array([[0, 1],
              [-2, -3]])
B = np.array([[0],
              [1]])
Q = np.array([[10, 0],
              [0, 1]])
R = np.array([[1]])

P = solve_continuous_are(A, B, Q, R)
K = inv(R) @ B.T @ P

print("LQR gain matrix K:", K)
