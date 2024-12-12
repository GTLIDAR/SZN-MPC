import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

from jax import random, jit, value_and_grad
from jax.config import config
from scipy.optimize import minimize
config.update("jax_enable_x64", True)



####### GP Functions #######


def kernel_A(X1, X2, theta, input_dim):
    """
    Anisotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m, d).
        X2: Array of n points (n, d).
        theta: kernel parameters (input_dim, )
    """
    
    #NOTE there should be equal number of "sqdist#" variables and input_dim
    
    sqdist0 = ((X1[:,0] ** 2).reshape(-1,1) + (X2[:,0] ** 2).reshape(1,-1) - 2 * X1[:,0].reshape(-1,1)@X2[:,0].reshape(1,-1))/(theta[0]**2)
    sqdist1 = ((X1[:,1] ** 2).reshape(-1,1) + (X2[:,1] ** 2).reshape(1,-1) - 2 * X1[:,1].reshape(-1,1)@X2[:,1].reshape(1,-1))/(theta[1]**2)
    sqdist2 = ((X1[:,2] ** 2).reshape(-1,1) + (X2[:,2] ** 2).reshape(1,-1) - 2 * X1[:,2].reshape(-1,1)@X2[:,2].reshape(1,-1))/(theta[2]**2)
    # sqdist3 = ((X1[:,3] ** 2).reshape(-1,1) + (X2[:,3] ** 2).reshape(1,-1) - 2 * X1[:,3].reshape(-1,1)@X2[:,3].reshape(1,-1))/(theta[3]**2)
    # sqdist4 = ((X1[:,4] ** 2).reshape(-1,1) + (X2[:,4] ** 2).reshape(1,-1) - 2 * X1[:,4].reshape(-1,1)@X2[:,4].reshape(1,-1))/(theta[4]**2)
    # sqdist5 = ((X1[:,5] ** 2).reshape(-1,1) + (X2[:,5] ** 2).reshape(1,-1) - 2 * X1[:,5].reshape(-1,1)@X2[:,5].reshape(1,-1))/(theta[5]**2)

    return theta[input_dim] ** 2 * jnp.exp(-0.5 * (sqdist0+sqdist1+sqdist2))

def jitter(d, value=1e-6):
    return jnp.eye(d) * value

def phi_opt_A(theta, X_m, X, y, sigma_y, input_dim):
  #theta = theta_fixed_A
  """Optimize mu_m and A_m using Equations (11) and (12)."""
  precision = (1.0 / sigma_y ** 2)

  K_mm = kernel_A(X_m, X_m, theta, input_dim) + jitter(X_m.shape[0])
  K_mm_inv = jnp.linalg.inv(K_mm)
  K_nm = kernel_A(X, X_m, theta, input_dim)
  K_mn = K_nm.T
    
  Sigma = jnp.linalg.inv(K_mm + precision * K_mn @ K_nm)
    
  mu_m = precision * (K_mm @ Sigma @ K_mn).dot(y)
  A_m = K_mm @ Sigma @ K_mm    
  
  return mu_m, A_m, K_mm_inv


def q_A(X_test, theta, X_m, mu_m, A_m, K_mm_inv, input_dim):
  """
  Approximate posterior. 
    
  Computes mean and covariance of latent 
  function values at test inputs X_test.
  """
  #theta = theta_fixed_A
  K_ss = kernel_A(X_test, X_test, theta, input_dim)
  K_sm = kernel_A(X_test, X_m, theta, input_dim)
  K_ms = K_sm.T

  f_q = (K_sm @ K_mm_inv).dot(mu_m)
  f_q_cov = K_ss - K_sm @ K_mm_inv @ K_ms + K_sm @ K_mm_inv @ A_m @ K_mm_inv @ K_ms
    
  return f_q, f_q_cov


############# MPC_GP Class ##############

class MPC_GP:
    
    # Constructor Method
    def __init__(self, all_csvFileName):
        '''
        FORMAT:
        all_csvFileName = [ ["mpc_state1.csv", "mpc_feedback1.csv"]  , ["mpc_state2.csv", "mpc_feedback2.csv"] , ... ]
        '''
        
        #Initialize training data from csv files
        vel_input = np.array([])
        dheading_input = np.array([])
        curr_vel_input = np.array([])
        errorx_output = np.array([])
        errory_output = np.array([])
       
        
        for pair_filenames in all_csvFileName:
            xd_com, dtheta, x, y, x_fb, y_fb, xd_fb = self.readfile(pair_filenames[0], pair_filenames[1])
            
            vel_input = np.append(vel_input, xd_com)
            dheading_input = np.append(dheading_input, dtheta)
            curr_vel_input = np.append(curr_vel_input, xd_fb)
            
            errorx_output = np.append(errorx_output, x - x_fb)
            errory_output = np.append(errory_output, y - y_fb)
            
        '''
        initialize Class variable
        '''
        self.input_dim = 3
        self.training_size = vel_input.size
        self.m = 60 #inducing variable size
        print("Training Data Size: ", self.training_size)
        print("Current Size of Inducing Variable: ", self.m)
        
        if self.m > self.training_size:
            raise Exception("Size of inducing variables must be smaller than the training data size.")
        
        #GP hyperparameters
        self.noiseStdA  = 1
        self.var_controller = 1
        self.theta_fixed_A = jnp.array([1,1,1, self.var_controller])
        self.noise_std_A_jax = jnp.array([self.noiseStdA])
        self.sigma_y = 0.2
        
        '''
        Define input training data
        '''
    
        self.X1 = np.empty((self.input_dim, self.training_size))
        
        self.X1[0] = vel_input
        self.X1[1] = dheading_input
        self.X1[2] = curr_vel_input

        self.X2 = self.X1

        '''
        Define output training data
        '''

        self.y1 = errorx_output
        self.y2 = errory_output
        
        #Run Optimization
        # Initialize inducing inputs
        indices = jnp.floor(jnp.linspace(0,self.X1.shape[1]-1,self.m)).astype(int)
        X_m = jnp.array(self.X1.T[indices,:])
        # res = minimize(fun=nlb_fn(X1.T, y1.T, sigma_y),
        #                 x0=pack_params(jnp.array(theta_fixed_A), X_m),
        #                 method='L-BFGS-B',
        #                 jac=True)
        res = minimize(fun=self.nlb_fn_A(self.X1.reshape(-1,self.input_dim), self.y1.reshape(-1,1)),
                        x0=self.pack_params_A(self.theta_fixed_A, self.noise_std_A_jax, X_m),
                        method='L-BFGS-B',
                        jac=True)

        self.theta_opt, self.sigma_y_opt, self.X_m_opt = self.unpack_params_A(res.x)
        self.mu_m_opt, self.A_m_opt, self.K_mm_inv = phi_opt_A(self.theta_opt, self.X_m_opt, self.X1.reshape(-1,self.input_dim), self.y1.reshape(-1,1), self.sigma_y_opt, self.input_dim)



        #Run Optimization
        # Initialize inducing inputs
        indices = jnp.floor(jnp.linspace(0,self.X2.shape[1]-1,self.m)).astype(int)
        X_m2 = jnp.array(self.X2.T[indices,:])
        # res = minimize(fun=nlb_fn(X1.T, y1.T, sigma_y),
        #                 x0=pack_params(jnp.array(theta_fixed_A), X_m),
        #                 method='L-BFGS-B',
        #                 jac=True)

        res2 = minimize(fun=self.nlb_fn_A(self.X2.reshape(-1,self.input_dim), self.y2.reshape(-1,1)),
                        x0=self.pack_params_A(self.theta_fixed_A, self.noise_std_A_jax, X_m2),
                        method='L-BFGS-B',
                        jac=True)

        self.theta_opt2, self.sigma_y_opt2, self.X_m_opt2 = self.unpack_params_A(res2.x)
        self.mu_m_opt2, self.A_m_opt2, self.K_mm_inv2 = phi_opt_A(self.theta_opt2, self.X_m_opt2, self.X2.reshape(-1,self.input_dim), self.y2.reshape(-1,1), self.sigma_y_opt2, self.input_dim)       
        
        print("GP Models are trained!")
        
    #####################
    # 
    # extract data functions 
    # 
    # #####################
    
    def readfile(self, filename_mpcState, filename_mpcFeedback):
        
        '''
        Load file from mpc_state/[]...]
        '''

        with open(filename_mpcState, "r") as f:
            allData = f.read().split("\n")
            
            #get all headers
            Header = allData[0].split(",.")
            # print("variables name: ", Header)
            
            #find index of the specific headers we want to retrieve their data
            # i_header_seq = Header.index("header.seq")
            # i_time = Header.index("time")
            i_dtheta = Header.index("dtheta")
            i_xd_com = Header.index("xd_com")
            i_x = Header.index("x")
            i_y = Header.index("y")
            
            # initialize array
            # header_seq = []
            # time = []
            dtheta = []
            xd_com = []
            x = []
            y = []

            
            lcv = 0
            for line in allData:
                
                if lcv == 0:
                    lcv += 1
                    continue
                
                if lcv == len(allData) - 1:
                    break
                
                line = line.replace(", " , ";")
                
                array = line.split(",")
                
                # header_seq.append(float(array[i_header_seq]))
                # time.append(array[i_time])
                dtheta.append(float(array[i_dtheta]))
                xd_com.append(float(array[i_xd_com]))
                x.append(float(array[i_x]))
                y.append(float(array[i_y]))

                #update loop control variable
                lcv += 1

        #truncate array and convert to numpy array    
        startMoving  = 0 #cutoff index

        # header_seq = np.array(header_seq[startMoving:-1])
        # time = np.array(time[startMoving:-1])
        dtheta = np.array(dtheta[startMoving:-1])
        xd_com = np.array(xd_com[startMoving:-1])
        x = np.array(x[startMoving:-1])
        y = np.array(y[startMoving:-1])

        # Convert string timestamps to datetime objects
        # timestamps = [datetime.strptime(ts, '%Y/%m/%d/%H:%M:%S.%f') for ts in time]

        #-----------------------------------------------------------------------------------------------------

        '''
        Load file from mpc_feedback/[]...]
        '''

        with open(filename_mpcFeedback, "r") as f:
            allData = f.read().split("\n")
            
            #get all headers
            Header = allData[0].split(",.")
            # print("variables name: ", Header)
            
            #find index of the specific headers we want to retrieve their data
            # i_time = Header.index("time")
            i_x = Header.index("x")
            i_y = Header.index("y")

            i_xd = Header.index("xd_com")

            
            
            # initialize array
            # time_fb = []
            x_fb = [] #fb for feedback
            y_fb = []
            xd_fb = []

            
            lcv = 0
            for line in allData:
                
                if lcv == 0:
                    lcv += 1
                    continue
                
                if lcv == len(allData) - 1:
                    break
                
                line = line.replace(",)" , ")")
                array = line.split(",")
                
                # time_fb.append(array[i_time])
                x_fb.append(float(array[i_x].replace('"(',"").replace(')"',"")))
                y_fb.append(float(array[i_y].replace('"(',"").replace(')"',"")))
                xd_fb.append(float(array[i_xd].replace('"(',"").replace(')"',"")))
                
                #update loop control variable
                lcv += 1

        #truncate array and convert to numpy array    
        startMoving  = 0 #cutoff index

        # time_fb = np.array(time[startMoving:-1])
        x_fb = np.array(x_fb[startMoving:-1])
        y_fb = np.array(y_fb[startMoving:-1])
        xd_fb = np.array(xd_fb[startMoving:-1])

        # # Convert string timestamps to datetime objects
        # timestamps_fb = [datetime.strptime(ts, '%Y/%m/%d/%H:%M:%S.%f') for ts in time_fb]

        return xd_com, dtheta, x, y, x_fb, y_fb, xd_fb

        
    #####################
    # 
    # GP Functions 
    # 
    # #####################
    def kernel_A(self, X1, X2, theta):
        """
        Anisotropic squared exponential kernel.

        Args:
            X1: Array of m points (m, d).
            X2: Array of n points (n, d).
            theta: kernel parameters (7,)
        """
        sqdist0 = ((X1[:,0] ** 2).reshape(-1,1) + (X2[:,0] ** 2).reshape(1,-1) - 2 * X1[:,0].reshape(-1,1)@X2[:,0].reshape(1,-1))/(theta[0]**2)
        sqdist1 = ((X1[:,1] ** 2).reshape(-1,1) + (X2[:,1] ** 2).reshape(1,-1) - 2 * X1[:,1].reshape(-1,1)@X2[:,1].reshape(1,-1))/(theta[1]**2)
        sqdist2 = ((X1[:,2] ** 2).reshape(-1,1) + (X2[:,2] ** 2).reshape(1,-1) - 2 * X1[:,2].reshape(-1,1)@X2[:,2].reshape(1,-1))/(theta[2]**2)
        # sqdist3 = ((X1[:,3] ** 2).reshape(-1,1) + (X2[:,3] ** 2).reshape(1,-1) - 2 * X1[:,3].reshape(-1,1)@X2[:,3].reshape(1,-1))/(theta[3]**2)
        # sqdist4 = ((X1[:,4] ** 2).reshape(-1,1) + (X2[:,4] ** 2).reshape(1,-1) - 2 * X1[:,4].reshape(-1,1)@X2[:,4].reshape(1,-1))/(theta[4]**2)
        # sqdist5 = ((X1[:,5] ** 2).reshape(-1,1) + (X2[:,5] ** 2).reshape(1,-1) - 2 * X1[:,5].reshape(-1,1)@X2[:,5].reshape(1,-1))/(theta[5]**2)


        return theta[self.input_dim] ** 2 * jnp.exp(-0.5 * (sqdist0+sqdist1+sqdist2))


    def kernel_diag_A(self, d, theta):
        """
        Isotropic squared exponential kernel (computes diagonal elements only).
        """
        return jnp.full(shape=d, fill_value=theta[0:-1] ** 2)


    def jitter(self, d, value=1e-6):
        return jnp.eye(d) * value


    def softplus(self, X):
        return jnp.log(1 + jnp.exp(X))


    def softplus_inv(self, X):
        return jnp.log(jnp.exp(X) - 1)


    def pack_params_A(self, theta, sigma_y, X_m):
        return jnp.concatenate([self.softplus_inv(theta), self.softplus_inv(sigma_y), X_m.ravel()])

    def unpack_params_A(self, params):
        return self.softplus(params[:(self.input_dim+1)]), self.softplus(params[(self.input_dim+1)]), jnp.array(params[(self.input_dim + 2):].reshape(-1,self.input_dim))

    def nlb_fn_A(self, X, y):
        n = X.shape[1]

        def nlb(params):
            """
            Negative lower bound on log marginal likelihood.
            
            Args:
                params: kernel parameters `theta` and inducing inputs `X_m`
            """
            theta, sigma_y, X_m = self.unpack_params_A(params)
            K_mm = self.kernel_A(X_m, X_m, theta) + self.jitter(X_m.shape[0])
            K_mn = self.kernel_A(X_m, X, theta)

            L = jnp.linalg.cholesky(K_mm)  # m x m
            A = jsp.linalg.solve_triangular(L, K_mn, lower=True) / sigma_y # m x n        
            AAT = A @ A.T  # m x m
            B = jnp.eye(X_m.shape[0]) + AAT  # m x m
            LB = jnp.linalg.cholesky(B)  # m x m
            c = jsp.linalg.solve_triangular(LB, A.dot(y), lower=True) / sigma_y  # m x 1

            # Equation (13)
            lb = - n / 2 * jnp.log(2 * jnp.pi)
            lb -= jnp.sum(jnp.log(jnp.diag(LB)))
            lb -= n / 2 * jnp.log(sigma_y ** 2)
            lb -= 0.5 / sigma_y ** 2 * y.T.dot(y)
            lb += 0.5 * c.T.dot(c)
            lb -= 0.5 / sigma_y ** 2 * jnp.sum(self.kernel_diag_A(n, theta))
            lb += 0.5 * jnp.trace(AAT)

            return -lb[0, 0]

        # nlb_grad returns the negative lower bound and 
        # its gradient w.r.t. params i.e. theta and X_m.
        nlb_grad = jit(value_and_grad(nlb))

        def nlb_grad_wrapper(params):
            value, grads = nlb_grad(params)
            # scipy.optimize.minimize cannot handle
            # JAX DeviceArray directly. a conversion
            # to Numpy ndarray is needed.
            return np.array(value), np.array(grads)

        return nlb_grad_wrapper
        
    def predict_mpc_deviation(self, X_test):
        if X_test.shape[0] != self.input_dim:
            raise Exception("Test point must have a dimension of (" + self.input_dim + ",1).")
            
        errorX_pred, errorX_cov = q_A(X_test.T, self.theta_opt, self.X_m_opt, self.mu_m_opt, self.A_m_opt, self.K_mm_inv, self.input_dim)
        errorY_pred, errorY_cov = q_A(X_test.T, self.theta_opt2, self.X_m_opt2, self.mu_m_opt2, self.A_m_opt2, self.K_mm_inv2, self.input_dim)
        
        return errorX_pred, errorX_cov, errorY_pred, errorY_cov


# all_csvFileName = [["2023-12-06-15-25-mpc_state.csv","2023-12-06-15-25-mpc_feedback.csv"],
#                    ["2023-12-06-18-40-mpc_state.csv","2023-12-06-18-40-mpc_feedback.csv"],
#                    ["2023-12-07-12-16-mpc_state.csv","2023-12-07-12-16-mpc_feedback.csv"]]

# # Initialize and Train 2 GP models (one for errorX, other for errorY)
# mpc_gp = MPC_GP(all_csvFileName)


# X_test = np.array([[0.3], [0.0], [0.2]])

# #call GP function to predict the deviations
# errorX_pred, errorX_cov, errorY_pred, errorY_cov = mpc_gp.predict_mpc_deviation(X_test)

# print("velocity:", X_test[0])
# print("curr vel:", X_test[2])
# print("heading change:", X_test[1])
# print("----------------------")
# print("predicted error in x:", errorX_pred)
# print("predicted error in y:", errorY_pred)
