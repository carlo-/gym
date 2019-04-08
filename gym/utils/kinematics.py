import numpy as np
from mujoco_py import functions as mj_fns


def get_jacobian(model, data, bodyid):
    pos = np.zeros((3, model.nv))
    rot = np.zeros((3, model.nv))
    mj_fns.mj_kinematics(model, data)
    mj_fns.mj_comPos(model, data)
    mj_fns.mj_jacBody(model, data, pos.reshape(-1), rot.reshape(-1), bodyid)
    return np.concatenate((pos, rot))


def solve_qp_ik_vel(vel, jac, joint_pos, joint_lims=None, duration=None, margin=0.2, solver=None):
    """
    Solves the IK for a given pusher velocity using a QP solver, imposing joint limits.
    If the solution is optimal, it is guaranteed that the resulting joint velocities will not
    cause the joints to reach their limits (minus the margin) in the specified duration of time
    :param vel: desired EE velocity
    :param jac: jacobian
    :param joint_pos: current joint positions
    :param joint_lims: matrix of joint limits; if None, limits are not imposed
    :param duration: how long the specified velocity will be kept (in seconds); if None, 2.0 is used
    :param margin: maximum absolute distance to be kept from the joint limits
    :param solver: the name of the solver to be used
    :return: tuple with the solution (as a numpy array) and with a boolean indincating if the result is optimal or not
    :type vel: np.ndarray
    :type jac: np.ndarray
    :type joint_pos: np.ndarray
    :type joint_lims: np.ndarray
    :type duration: float
    :type margin: float
    :type solver: str
    :rtype: (np.ndarray, bool)
    """

    import cvxopt
    x_len = len(joint_pos)

    P = cvxopt.matrix(np.identity(x_len))
    A = cvxopt.matrix(jac)
    b = cvxopt.matrix(vel)
    q = cvxopt.matrix(np.zeros(x_len))

    if duration is None:
        duration = 2.

    if joint_lims is None:
        G, h = None, None
    else:
        G = duration * np.identity(x_len)
        h = np.zeros(x_len)
        for i in range(x_len):
            dist_up = abs(joint_lims[i, 1] - joint_pos[i])
            dist_lo = abs(joint_lims[i, 0] - joint_pos[i])
            if dist_up > dist_lo:
                # we are closer to the lower limit
                # => must bound negative angular velocity, i.e. G_ii < 0
                h[i] = dist_lo
                G[i, i] *= -1
            else:
                # we are closer to the upper limit
                # => must bound positive angular velocity, i.e. G_ii > 0
                h[i] = dist_up
        h = cvxopt.matrix(h - margin)
        G = cvxopt.matrix(G)

    if solver is None:
        qp_kwargs = dict(options=dict(show_progress=False))
    elif solver == 'ldl':
        qp_kwargs = dict(options=dict(show_progress=False, kktreg=1e-9), kktsolver='ldl')
    elif solver == 'mosek':
        import mosek
        qp_kwargs = dict(options=dict(show_progress=False, mosek={mosek.iparam.log: 0}), solver='mosek')
    else:
        raise NotImplementedError

    sol = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h, **qp_kwargs)

    x = np.array(sol['x']).reshape(-1)
    optimal = sol['status'] == 'optimal'

    return x, optimal
