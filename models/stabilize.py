import numpy as np
import matplotlib.pyplot as plt
import itertools
import cdd

def ackermann(A, B, poles, plot=False):
    """
    Computes the control input matrix K using Ackermann's formula.

    Parameters:
    - A: State matrix of the linear dynamical system
    - B: Input matrix of the linear dynamical system
    - poles: Desired eigenvalues (poles) of the closed-loop system

    Returns:
    - K: Control input matrix
    - Acl: Closed-loop system dynamics
    """

    print('\nStabilizing dynamics by placing poles at', poles)

    n = A.shape[0]  # Number of state variables

    # Check controllability of the system
    C = np.hstack([B] + [np.dot(np.linalg.matrix_power(A, i), B) for i in range(1, n)])
    if np.linalg.matrix_rank(C) != n:
        raise ValueError("System is not controllable.")

    # Compute the characteristic polynomial coefficients
    char_poly = np.poly(poles)
    
    # Compute Ackermann matrix
    PA = np.zeros((n, n))
    for i in range(n+1):
        PA += np.linalg.matrix_power(A, i) * char_poly[::-1][i]

    # Compute feedback gain
    zeros = np.zeros([1,n])
    zeros[0,-1] = 1
    Kgain = zeros @ np.linalg.inv(C) @ PA

    print('- Feedback gain is K = ', np.round(Kgain, 4))

    # Compute closed-loop system
    Acl = (A - B @ Kgain)
    print('- Closed-loop dynamics matrix:\n', Acl)
    assert np.linalg.matrix_rank(Acl) == n

    print('- Eigen values of the closed-loop dynamics:\n', np.round(np.linalg.eigvals(Acl), 4))

    if plot:
        # Validate if the system is stable indeed
        x = {}

        steps = 100

        x0 = [
            np.arange(-5, 5+1e-3),
            np.arange(-5, 5+1e-3)[::-1]
        ]


        # plot
        fig, ax = plt.subplots()

        for i, x0 in enumerate(itertools.product(*x0)):
            x0 = np.array(x0)
            
            x[i] = np.zeros((steps+1, n))

            x[i][0] = [x0[0], x0[1]]

            for j in range(steps):

                x[i][j+1] = Acl @ x[i][j]

            ax.plot(x[i][:,0], x[i][:,1], linewidth=2.0)

            break

        plt.show(block = False)
        
    return Acl, Kgain


def lqr(A, B, Q, R):

    import control as ct

    K,S,E = ct.dlqr(A, B, Q, R)
    print('Eigenvalues of closed-loop system:', E)
    print('Control gain matrix:', K)

    return K

def compute_stabilized_control_vertices(model, target_point):

    alpha = -model.K @ model.A_inv @ (target_point - model.Q_flat)
    beta  = np.eye(model.p) + model.K @ model.A_inv @ model.B

    G = np.vstack([beta, -beta, np.eye(model.p), -np.eye(model.p)])
    h = np.hstack([model.uMax - alpha, -model.uMin + alpha, model.uBarMax, -model.uBarMin])

    # Get vertices of feasible control input set
    # cddlib wants the input Ax <= b as a single matrix [b -A]
    M = np.c_[h, -G]

    # build cdd matrix, set representation to inequalities
    mat = cdd.Matrix(M, number_type='float')
    mat.rep_type = cdd.RepType.INEQUALITY
    
    # build polyhedron
    poly = cdd.Polyhedron(mat)
    
    # get vertices
    gen = np.array(poly.get_generators())

    if len(gen) > 0:
        u_vertices = np.array(gen)[:,1:]
    else:
        u_vertices = np.array([])

    return u_vertices