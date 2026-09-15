import numpy as np

def bisection_solver(f, a, b, tol):
    """ Approximates the root of f bounded by a and b within tolerance
    |f(m)| < tol with m the midpoint between a and b recursive implementation.

    Args:
        f (function): a univariate function to find the root
        a (float): lower bound to search
        b (float): upper bound to search
        tol (float): tolerance
    """

    # Check if a and b bound a root
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The sclars a and b do not bound a root")
    
    # get midpoint
    m = (a + b) / 2

    if np.abs(f(m)) < tol:
        # Stopping condition, report m as a root
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        # Case where m is an improvement on a.
        # Make a recursive call with a = m
        return bisection_solver(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        # Case where m is an improvement on b.
        # Make recursive call with b = m
        return bisection_solver(f, a, m, tol)
    

if __name__ == "__main__":
    f = lambda x: x ** 2 - 2

    r1 = bisection_solver(f, 0, 2, 0.1)
    print("r1 =", r1)
    r01 = bisection_solver(f, 0, 2, 0.01)
    print("r01 =", r01)

    print("f(r1) =", f(r1))
    print("f(r01) =", f(r01))
