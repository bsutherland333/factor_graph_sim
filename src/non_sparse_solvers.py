import numpy as np


def numpy_lstsq(A, b, x):
    """
    Solve a linear system of equations using numpy's least-squares solver.

    Parameters:
    A (np.array): The collected and whitened Jacobian matrix.
    b (np.array): The collected and whitened residual vector

    Returns:
    np.array: The solution to the linear system.
    """

    return np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1, x.shape[1]) + x


def cholesky_factorization(A, b, x):
    """
    Perform a Cholesky factorization to solve a linear system of equations.

    Note that this method requires that the matrix A is positive definite, making it a
    bit of a pain to use sometime.

    Parameters:
    A (np.array): The collected and whitened Jacobian matrix.
    b (np.array): The collected and whitened residual vector
    x (np.array): The initial guess for the solution.

    Returns:
    np.array: The solution to the linear system.
    """

    information_matrix = A.T @ A
    R = np.linalg.cholesky(information_matrix).T
    y = np.linalg.solve(R.T, A.T @ b)
    delta = np.linalg.solve(R, y)
    x += delta.reshape(-1, x.shape[1])

    return x

