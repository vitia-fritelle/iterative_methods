import sys
import numpy as np

def gradient_method_wrapper(A: np.ndarray, b: np.ndarray, x: np.ndarray,
    tol: float = 0.1**5) -> np.ndarray:
    '''
        Gradient Method - Wrapper to the real function
        A: Matrix
        b: output array
        x: solution array
        tol: tolerance of the solution
    '''
    r: np.ndarray = b - A.dot(x)
    delta: float = r.dot(r)
    s: float = delta/(r.dot(A.dot(r)))
    x = x + s*r
    if np.sqrt(delta) < tol:
        return x
    else:
        return gradient_method_wrapper(A, b, x, tol)

def gradient_method(A: np.ndarray, b: np.ndarray, max: int = 200,
    tol: float = 0.1**5) -> np.ndarray:
    '''
        Gradient Method
        A: Matrix
        b: output array
        max: maximum number of iterations
        tol: tolerance of the solution
    '''
    sys.setrecursionlimit(max)
    x = b/np.diag(A) #initial guess
    try:
        return gradient_method_wrapper(A, b, x, tol)
    except RuntimeError:
        print(f'Solution did not converge with {max} iterations.')
        return x

def conjugated_gradient_method_wrapper(A: np.ndarray, b: np.ndarray, 
    x: np.ndarray, r: np.ndarray, aux: float, v: np.ndarray,
    tol: float = 0.1**5) -> np.ndarray:
    '''
        Conjugated Gradient Method - Wrapper to the real function
        A: Matrix
        b: output array
        x: solution array
        r: residue array associated with x
        aux: residue value associated with x
        v: array that represents the direction to go
        tol: tolerance of the solution
    '''
    z: np.ndarray = A.dot(v)
    s: float = aux/z.dot(v)
    x = x + s*v
    r = r - s*z
    delta: float = r.dot(r)
    if np.sqrt(delta) < tol:
        print(np.sqrt(delta))
        return x
    else:
        m = delta/aux
        aux = delta
        v = r + m*v
        return conjugated_gradient_method_wrapper(A, b, x, r, aux, v, tol)

def conjugated_gradient_method(A: np.ndarray, b: np.ndarray,
    max: int = 200, tol: float = 0.1**5) -> np.ndarray:
    '''
        Conjugated Gradient Method
        A: Matrix
        b: output array
        max: maximum number of iterations
        tol: tolerance of the solution
    '''
    sys.setrecursionlimit(max)
    x = b/np.diag(A) #initial guess
    r = b - A.dot(x)
    aux = r.dot(r)
    v = b - A.dot(x)
    try:
        return conjugated_gradient_method_wrapper(A, b, x, r, aux, v, tol)
    except RuntimeError:
        print(f'Solution did not converge with {max} iterations.')
        return x

def create_matrix(i: int, j: int, n: int, r: int):
    '''
        Function to create a matrix element
        i: row of matrix
        j: column of matrix
        n: size of matrix
        r: bandwith of matrix
    '''
    if i == j:
        if i == 0 or i == n-1:
            return -2
        else:
            return -4
    elif j == i+1 or i == j+1:
        return 1
    elif j == i+r or i == j+r:
        return 1
    else:
        return 0

if __name__ == "__main__":
    r = 6 # matrix bandwidth
    n = 15 # matrix size
    A = np.array(
        [[create_matrix(i,j,n,r) for j in range(n)] for i in range(n)]
    )
    x = np.ones(n)
    x[-1] = -100
    b = A.dot(x)
    sol = conjugated_gradient_method(A, b)
    print(sol)

