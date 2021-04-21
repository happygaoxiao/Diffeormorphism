import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# numpy function for quaternion logarithm
def acos_np(v):
    # To ensure that the distance between two antipodal rotations is zero
    # refer to the paper 'An Approach for Imitation Learning on Riemannian Manifolds'
    sign = v < 0
    return np.arccos(v) - sign * np.pi


def log_q(q, eps=1e-15):
    # refer to "Orientation in Cartesian space dynamic movement primitives"
    # and "Filtering in a unit quaternion space for model-based object tracking"
    if q.ndim == 1:
        v = q[0]
        u = q[1:]
        u_norm = np.linalg.norm(u)
        r = np.array([0., 0, 0])
        if u_norm > eps:
            r = acos_np(v) * u / u_norm
    else:
        r = np.zeros([q.shape[0], 3])
        for i in range(q.shape[0]):
            v = q[i, 0]
            u = q[i, 1:]
            u_norm = np.linalg.norm(u)
            if u_norm > eps:
                r[i, :] = acos_np(v) * u / u_norm

    return r


# for tensorflow
def exp_tf(r, eps=1e-10):
    # input (3,) or (n, 3)
    if r.shape.ndims == 1:
        r_norm = tf.norm(r + eps)
        # q0 = tf.constant([1.,0,0,0])
        q1 = tf.concat([tf.cos(r_norm) * tf.constant([1.]), tf.sin(r_norm) * r / r_norm], axis=0)
        # q = tf.where(r_norm < eps, q0, q1 ) # if r_norm <  eps
        q = q1
        # if r_norm < eps:
        #     q = [1,0,0,0]
        # else:
        #     v = tf.cos(r_norm)
        #     u = tf.sin(r_norm) * r/r_norm
        #     q = tf.concat([v, u], 1)

    elif r.shape.ndims == 2:
        r_norm = tf.norm(r + eps, axis=1)
        # s_bool = tf.math.greater(r_norm, eps)
        v = tf.cos(r_norm)
        v = tf.expand_dims(v, 1)
        # u = tf.sin(r_norm)/r_norm * r
        # tmp = tf.expand_dims(r_norm, 1)
        # u = tmp * r              # [n,1] * [n,3]
        # q = tf.concat([v, u], 1)

        # q0 = tf.constant([1.0, 0, 0, 0]) and repeat
        # q0 = tf.concat([tf.ones_like(tmp),tf.zeros_like(r) ], 1)
        q1 = tf.concat([v, tf.expand_dims(tf.sin(r_norm) / r_norm, 1) * r], 1)
        # q = tf.where(r_norm < eps, q0, q1 ) # if r_norm <  eps
        q = q1
    else:
        raise NotImplementedError
    return q


def exp_np(r):
    if r.ndim == 1:  # input (3,)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-15:
            q = [1, 0, 0, 0]
        else:
            v = np.cos(r_norm)
            u = np.sin(r_norm) * r / r_norm
            q = np.hstack([v, u])
    elif r.ndim == 2:  # input (n,3)
        q_ = []
        for i in range(r.shape[0]):
            r_norm = np.linalg.norm(r[i, :])
            if r_norm < 1e-15:
                q_.append(np.array([1, 0, 0, 0]))
            else:
                v = np.cos(r_norm)
                u = np.sin(r_norm) * r[i, :] / r_norm
                q_.append(np.hstack([v, u]))
        q = np.array(q_)  # [n,4], [w,x,y,z] order
    else:
        raise NotImplementedError
    return q


# Metrics in SO3 for quaternions
def ori_dis_tf(q1, q2, normalize=False):
    # in Tensorflow 1.15
    # quaternion distance between q1 and q2 in SO3
    # q1  [None,4] tensor
    # q2  [None,4] tensor
    # return dis, [None ,]
    if q1.shape.ndims == 1:
        if normalize:
            q1, _ = tf.linalg.normalize(q1)
            q2, _ = tf.linalg.normalize(q2)
        angle = 2 * tf.reduce_sum(q1 * q2) ** 2 - 1
        # angle_clip = tf.clip_by_value(angle,-1,1)  # fix to [-1,1] for acos
        dis = tf.acos(angle)
    else:
        if normalize:
            q1, _ = tf.linalg.normalize(q1, axis=1)
            q2, _ = tf.linalg.normalize(q2, axis=1)
        angle = 2 * tf.reduce_sum(q1 * q2, axis=1) ** 2 - 1
        # angle_clip = tf.clip_by_value(angle,-1,1)  # fix to [-1,1] for acos, will lose gradient
        # dis = tf.acos(angle_clip )
        # angle_clip = tf.clip_by_value(angle, -1 + 1e-10, 1 - 1e-10)  # fix to [-1,1] for acos
        angle_clip = angle - tf.math.sign(angle) * 1e-8
        # dis =  tf.atan2(x_sqrt, x) # still not working, got a nan error of the cost function
        dis =  tf.acos(angle_clip)
    return dis


def dis_tan_tf(r1, r2):
    # r1 [None, 3] tangent space tensor
    # r2 [None, 3] tangent space tensor
    # output distance in SO3
    q1 = exp_tf(r1)
    q2 = exp_tf(r2)
    return ori_dis_tf(q1, q2)


def ori_dis_np(q1, q2):
    # quaternion distance between q1 and q2
    # q1  [n x 4] or [4,] numpy array
    # q2  [n x 4] or [4,] numpy array
    # return [n, ] distance numpy array, in [0, pi]
    if len(q1.shape) == 1:
        dis = np.arccos(2 * np.sum(q1 * q2) ** 2 - 1)
    else:
        tmp = 2 * np.sum(q1 * q2, axis=1) ** 2 - 1
        tmp = np.clip(tmp, -1, 1)  # adjust to [-1,1] if out of range

        # Because the rounding error from computation,
        # sometimes tmp may be a little bigger than 1.
        dis = np.arccos(tmp)
    return dis


# generate data:


# Visualization for 3D plot:
def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_aspect('equal')
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
