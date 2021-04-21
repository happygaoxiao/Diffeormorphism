import numpy as np
import quaternion as qua # [w,x,y,z] order
import time
from scipy.optimize import minimize
import scipy.optimize as sci_opti



class iteration(object):
    def __init__(self, para, x_data,y_data, training = True, position_mapping=True):
        # para: [k, rho, beta]
        # x_data: [n,7]
        # y_data: [n,7]
        self.k = int(para[0])
        self.nums = x_data.shape[0]
        self.nbState = x_data.shape[1] - 4  # dimension of position
        self.para = para
        self.x_data = np.copy(x_data)
        self.y_data = np.copy(y_data)
        self.position_mapping = position_mapping
        if training:
            start_time = time.time()
            self.learnt_data = self.mapping(self.x_data, self.y_data,self.para)
            print 'Number of points =',self.nums,'in',self.nbState,'D space.' 
            print 'iteration number =',self.k
            print 'training time', time.time() - start_time, '[s].'
            self.mapping_error(angle_degree=True)
        
    def mapping(self, x, y, para):
        num_data = x.shape[0]
        # initialization
        k = self.k
        N = self.nbState
        rho_belta = np.zeros((k,2))
        p = np.zeros((k,N))   #  position transform center
        v = np.zeros((k,N))   #  transform vector

        rho_belta_ori = np.zeros((k,2)) # rho for RBF function, belta
        p_ori = np.zeros((k,N))   #  orientation transform center
        v_q = np.repeat(np.quaternion(1,0,0,0),k) #  orientation change 
        # cost = np.zeros((k,N))

        xi = np.copy(x)
        for i in range(k):
            if self.position_mapping:
            # position iteration
                m = np.argmax(np.sum((xi[:,:N] - y[:,:N])**2, axis=1))
                p[i,:] = xi[m,:N]
                
                q = y[m,:N]
                v0 = (q - p[i,:])
                # v0 = para[1]*(q - p[i,:])  # translation vector with the max distance, [2,]
                norm_v0 = np.sqrt(np.sum(v0**2))
                up_bound = self.para[1] *  np.sqrt(np.exp(1.)/2)/norm_v0 # for rho upbound to keep the diffeomorphism.
                x0 = np.array([[up_bound/10,0.5]])                 # initial values for [rho, belta]
                bnds = (0,up_bound),(0.9,0.9)  # rho, belta bounds
                args = (xi[:,:N], v0, p[i,:], y[:,:N], num_data)
                res_p = minimize(self.pos_cost_fun,x0,args,bounds=bnds) # solve the 2-parameter minimum problem
                rho_belta[i,:] = res_p.x
                v[i,:] = rho_belta[i,1] * v0
                xi[:,:N] = xi[:,:N] + np.exp(-rho_belta[i,0]**2 * np.sum((xi[:,:N] - p[i,:])**2,axis=1).reshape(-1,1))* v[i,:].reshape(1,-1)

    #         # orientation iteration
            n = np.argmax(  self.ori_dis(xi[:,N:], y[:,N:])  )
            p_ori[i,:] = xi[n, :N]  # center
            v_q0 = qua.from_float_array(y[n,N:])/ qua.from_float_array(xi[n,N:])  # max ori difference
            bnds = [(0,20),(0.9 ,0.9)]  # rho, belta bounds 
            args = (xi, v_q0, p_ori[i,:], y, num_data)
            x0 = np.array([[0.5, 0.5]]) 
            res_q = minimize(self.ori_cost_fun, x0, args, bounds=bnds, method='SLSQP') # solve the 2-parameter minimum problem
            rho_belta_ori[i,:] = res_q.x        
            v_q[i] = qua.slerp(np.quaternion(1,0,0,0), v_q0, 0, 1, rho_belta_ori[i,1]) 
            weights = np.exp(  -rho_belta_ori[i,0]**2 * np.sum((xi[:,:N] - p_ori[i,:])**2,axis=1)  ) # \in (0,1]
            q_slerp = qua.slerp(np.quaternion(1,0,0,0), v_q[i], 0, 1, weights)       # Spherical linear interpolation
            xi_q_new = q_slerp * qua.from_float_array( xi[:,N:] )
            #update 
            xi[:,N:] = qua.as_float_array(xi_q_new)
            # xi[:,:N] = xi[:,:N] + np.exp(-rho_belta[i,0]**2 * np.sum((xi[:,:N] - p[i,:])**2,axis=1).reshape(-1,1))* v[i,:].reshape(1,-1)

        learnt_data = [rho_belta, p,v, rho_belta_ori, p_ori, v_q]
        return learnt_data 

    def forward(self, x, ori=True, Jac=False):
        ### input:
        #           x: [n,7],  [x,y,z,qw,qx,qy,qz]
        ### output:
        #           y: [n,7],  [x,y,z,qw,qx,qy,qz]
        #           J:  [n, n], Jacobian of forward mapping
        rho_belta, p, v = self.learnt_data[0], self.learnt_data[1], self.learnt_data[2]  # position data
        rho_belta_ori, p_ori, v_q = self.learnt_data[3], self.learnt_data[4], self.learnt_data[5]  # orientation data
        k = self.k
        N = self.nbState
        if x.ndim ==1:
            x = x.reshape(1,-1)
        y = np.copy(x)
        if Jac:
            J = np.identity(N)      
        # y_steps = []
        for i in range(k):
            if Jac:
                tmp =  np.exp(-rho_belta[i,0]**2 * np.sum((y[0,:N] - p[i,:])**2)) * (-2) * rho_belta[i,0]**2*(y[0,:N]-p[i,:]).reshape(1,-1)
                J_i = np.identity(N) + np.dot(v[i,:].reshape(-1,1), tmp)
                J = np.dot(J_i, J)
            if self.position_mapping:
                y[:,:N] = y[:,:N] + np.exp(-rho_belta[i,0]**2 * np.sum((y[:,:N] - p[i,:])**2,axis=1).reshape(-1,1))* v[i,:].reshape(1,-1)
            # y_steps.append(np.copy(y[:,:2]))
            # orientation
            if ori:
                weights = np.exp(  -rho_belta_ori[i,0]**2 * np.sum((y[:,:N] - p_ori[i,:])**2,axis=1)  ) # \in (0,1]
                q_slerp = qua.slerp(np.quaternion(1,0,0,0), v_q[i], 0, 1, weights)       # Spherical linear interpolation
                xi_q_new = q_slerp * qua.from_float_array( y[:,N:] ) 
                y[:,N:] = qua.as_float_array(xi_q_new)
        if Jac:
            return y, J
        else:
            return y
        
    def f_backward_position(self, x, *args):
        y, rho, v, p = args[0], args[1], args[2], args[3]
        Eq = x + v * np.exp(-rho ** 2 * np.sum((x - p) ** 2, axis=0)) - y
        return Eq


        # the backward_evaluation is done by Newton's method
    def backward(self, y, ori=True):
        # y:  (7,) numpy
        # x:  return (7,), [position, qua_wxyz],numpy array
        if y.ndim != 1:
            y = y.flatten()
            # raise NotImplementedError
        rho_belta, p, v = self.learnt_data[0], self.learnt_data[1], self.learnt_data[2]  # position data
        rho_belta_ori, p_ori, v_q = self.learnt_data[3], self.learnt_data[4], self.learnt_data[5]  # orientation data
        N = self.nbState
        x = np.copy(y)
        qx = qua.from_float_array(np.copy(y[N:]))
        k = self.k
        for i in range(k - 1, -1, -1):
            # for orientation
            if ori:
                weights = np.exp(-rho_belta_ori[i, 0] ** 2 * np.sum((x[:N] - p_ori[i, :]) ** 2))        
                # vi = v_q[i]
                q_slerp = qua.slerp(np.quaternion(1, 0, 0, 0), v_q[i], 0, 1,
                                            weights)  # Spherical linear interpolation
                # qx = (vi ** weights).conj() * qx
                # print qx
                qx = q_slerp.conj() * qx
                x[N:] =  qua.as_float_array(qx)
            if self.position_mapping:
                args = (x[:N], rho_belta[i, 0], v[i, :], p[i, :])
                x[:N] = sci_opti.fsolve(self.f_backward_position, x0=x[:N], args=args)
            
        return x  # shape (7,)


    def pos_cost_fun(self,x, *args):
        xi, v0, p0, y, n = args[0],args[1],args[2],args[3],args[4]
        v0 = v0.reshape(1, -1)
        p0 = p0.reshape(1, -1)
        x22 = xi + x[1]*np.repeat(v0,n,axis=0)* np.exp(-x[0]**2 * np.sum((xi - p0)**2,axis=1).reshape(-1,1))
        dis = np.sum( np.sqrt(np.sum((x22-y)**2, axis=1)) )/n
        return dis


    def ori_cost_fun(self, x, *args):
        N = self.nbState 
        xi_all, v_q0, p_ori0, y_all, num_data = args[0],args[1],args[2],args[3],args[4]
        q_identity = np.quaternion(1,0,0,0)
        weights = np.exp(  -x[0]**2 * np.sum((xi_all[:,:N] - p_ori0)**2,axis=1)  ) # \in (0,1]
        q_max = qua.slerp(q_identity, v_q0, 0, 1, x[1]) 
        q_slerp = qua.slerp(q_identity, q_max, 0, 1, weights)       # Spherical linear interpolation
        # for i in range(num_data):
        #     if q_slerp[i].isnan():
        #         print 'q_slerp is nan'
        xi_q_new = q_slerp * qua.from_float_array( xi_all[:,N:] ) 
        xi_new_np = qua.as_float_array(xi_q_new)
        # if np.sum(np.isnan(xi_new_np)):
        #     print 'nan'
        dis = self.ori_dis(xi_new_np , y_all[:,N:] ) # new ori dis
        return np.sum(dis)/num_data

    def ori_dis(self, q1, q2):
        # qua distance between q1 and q2
        # q1  [n x 4] or [4,] numpy array
        # q2  [n x 4] or [4,] numpy array
        # return [n, ] distance numpy array, in [0, pi]
        eps = 1e-12
        if len(q1.shape) == 1:
            tmp = np.clip(np.arccos( 2* np.sum(q1*q2)**2 -1)  ,eps,1-eps ) # adjust tmp to [0,1] if out of range
            dis = tmp
        else:
            tmp = 2* np.sum(q1*q2, axis=1)**2 -1
            tmp = np.clip(tmp,eps,1-eps ) # adjust tmp to [0,1] if out of range
    #        for i in range(tmp.shape[0]):
    #            if tmp[i]>1 or tmp[i]<0:
    #                print 'tmp out of range', tmp[i]
    # Because the rounding error from computation,
    # sometimes tmp may be a little bigger than 1.
            dis = np.arccos( tmp)  
        return dis   
    
    def mapping_error(self, angle_degree = False):

        N = self.nbState
        y_pred = self.forward(self.x_data)

        error = np.sqrt(np.sum((y_pred[:,:N]- self.y_data[:,:N])**2,axis=1))
        print '######### Estimation'
        print 'Total pos error mean+std:', np.mean(error), np.std(error), "[m]"
        # print 'End point position error:', error[-1], '[m]'
        dis = self.ori_dis( y_pred[:,N:], self.y_data[:,N:])
        if angle_degree:
            dis = dis*180/np.pi
            print 'Total ori error mean+std:', np.mean(dis ), np.std(dis ), '[degree]'
            # print 'End point ori error:', dis[-1], '[degree]'
        else:
            print 'Total ori error mean+std:', np.mean(dis ), np.std(dis ), '[rad]'
            # print 'End point ori error:', dis[-1], '[rad]'
