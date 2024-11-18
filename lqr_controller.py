import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.linalg import inv

class LQRSolver:
    def __init__(self, params: dict):
        self.L = params['L']
        self.m = params['mass']
        self.Cy = params['Cy']
        self.Iz = params['Iz']
        self.B = np.array([[0], [params['Cy']/params['mass']], [0], [params['L']*params['Cy']/(2*params['Iz'])]])
        self.Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.R = np.array([[1]])

    def LQR_solve(self, v_B_x):
        L = self.L
        Cy = self.Cy
        m = self.m
        Iz = self.Iz
        A = np.array([[0, 1, 0, 0],
                      [0, -Cy/(m*v_B_x), Cy/m, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, -L*L*Cy/(2*Iz*v_B_x)]])
        P = solve_continuous_are(A, self.B, self.Q, self.R)
        K = inv(self.R) @ self.B.T @ P
        return K

