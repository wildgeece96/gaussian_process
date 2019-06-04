import numpy as np 

class Kernel(object):
    def __init__(self, theta=[], seed=42):
        np.random.seed(seed)
        if len(theta) != 4:
            self.theta = np.random.randn(4)  
        else:
            self.theta = np.array(theta)  
    
    def make_kernel(self, X):
        """
        inputs : 
            X : 2d-array. shape=[N, d].
        returns : 
            K : 2d-array. shape=[N,N]. 
                The matrix that represents covariance.
        """
        N = X.shape[0] 
        
        # make kernel matrix 
        K = np.zeros(shape=[N,N],dtype=float)
        for i in range(N):
            for j in range(N):
                K[i,j] = self._kernel(X[i], X[j], i, j)  
        return K 
                
    def _kernel(self, x, x_, n=0, n_=1):
        """
        inputs : 
            x : 1d-array. shape=[d]
            x_ : 1d-array. shape=[d]
            n, n_ : the index of data.
        returns: 
            k : scaler.
        """
        k_1 = self.theta[0] * np.exp(- np.sum((x - x_)**2) / self.theta[1]) 
        k_2 = self.theta[2] * np.sum((x - x_)**2)  
        k_3 = self.theta[3] * (n==n_) # delta function 
        return k_1 + k_2 + k_3 
    
    def part_k(self, x, x_, n, n_):
        """
        inputs: 
           x, x_ : 1d-array. shape=[d]
           n, n_ : the index of data
        returns: 
            dk : 1d-array. shape=[4].
                the result of dk/dθ
        """
        dk = np.zeros(4)
        dk[0] =  np.exp(- np.sum((x - x_)**2) / self.theta[1])   
        dk[1] = self.theta[0] * np.sum((x-x_)**2) / self.theta[1]**2 * np.exp(- np.sum((x - x_)**2) / self.theta[1])  
        dk[2] = np.sum((x - x_)**2)
        dk[3] = 1.0 * (n==n_)  # delta function
        return dk 
    
    def part_K(self, X):
        """
        inputs: 
           X : 2d-array. shape=[N, d].
        returns: 
           dK :3d-array. shape=[4, N, N,].
               the result of dK/dθ
        """
        N = X.shape[0]
        dK = np.zeros(shape=[N, N, 4])
        for i in range(N):
            for j in range(N):
                dK[i, j] =self. part_k(X[i], X[j], i, j)
        dK = dK.transpose(2, 0, 1)
        return dK
    
    def make_k_star(self, X, X_star):
        """
        inputs: 
            X : 2d-array.  The known values
            X_star: 2d-array. shape=[M,d]. The unkonwn values.
        returns: 
            k_star: 2d-array. shape=[N, M]
        """
        N = X.shape[0]
        M = X_star.shape[0]
        k_star = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                k_star[i,j] = self._kernel(X[i], X_star[j])
        return k_star 
        
                
        
            