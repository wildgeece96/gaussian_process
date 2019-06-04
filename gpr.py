import numpy as np 

class GPRegression(object):
    
    def __init__(self,kernel, thetas=[]):
        """
        inputs: 
            kernel : the class for kernel.
        """
        self.kernel = kernel(thetas)
        
    def fit(self, X, y, num_iter=10, eta=0.1):
        """
        In this process, optimize the hyper parameters.
        inputs: 
            X : 2d-array. shape=[N, d]
                The explanatory variables.
            y : 1d-array. shape=[N]
                The objective variables.
        returns: 
            None
        """
        print("fiting ......")
        self.X = X  
        self.y = y 
        
        # optimize thetas 
        for i in range(num_iter):
            print("num_iter = {}".format(i), end="\r")
            dl = self.part_l() 
            self.kernel.theta += eta * dl  
        self.K = self.kernel.make_kernel(X) # kernel_matrix for predict to use for predicting value.
        print("fitting completed")
            
    def part_l(self):
        N = self.X.shape[0]
        k_theta = self.kernel.make_kernel(self.X)
        k_theta_inv = np.linalg.inv(k_theta)
        dk_dtheta = self.kernel.part_K(self.X)
        dl_1 = - np.trace(np.dot(dk_dtheta, k_theta_inv), axis1=1, axis2=2)
        dl_2_1 = (np.dot(k_theta_inv,self.y.reshape(-1,1))) 
        dl_2 = np.dot(np.dot(dl_2_1.T, dk_dtheta).reshape(4,N), dl_2_1) # -> [4,1]
        dl_2 = dl_2.flatten()
        return dl_1 + dl_2
    
    def predict(self, X_star):
        k_star = self.kernel.make_k_star(self.X, X_star)
        k_dstar = self.kernel.make_kernel(X_star)  
        K_inv = np.linalg.inv(self.K)
        m = np.dot(np.dot(k_star.T, K_inv), self.y.reshape(-1,1))
        v = k_dstar - np.dot(np.dot(k_star.T, K_inv), k_star)
        return m, v
    
    