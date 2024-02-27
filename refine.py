
import numpy as np


class UC_Model:
    def __init__(self,shape,n_peaks):
        self.shape = shape
        self.n_peaks = n_peaks
        
    def model(self,X,*params):
        
        x,y=X
        p = np.array(params).reshape((self.n_peaks,4))

        return np.sum([self.g(x,y,*pl) for pl in p],axis=0)
    
    def g(self, x,y,A,x0,y0,s):
        
        sy,sx = self.shape
        return A*np.exp(-((x/sx-x0)**2+(y/sy-y0)**2)/s**2)


class UC_Model_fix_sigma:
    def __init__(self,shape,n_peaks,sigmas):
        self.shape = shape
        self.n_peaks = n_peaks
        self.sigmas = sigmas
        
    def model(self,X,*params):
        
        x,y=X
        p = np.array(params).reshape((self.n_peaks,3))
        p = np.hstack([p,self.sigmas[:,np.newaxis]])


        return np.sum([self.g(x,y,*pl) for pl in p],axis=0)
    
    def g(self, x,y,A,x0,y0,s):
        
        sy,sx = self.shape
        return A*np.exp(-((x/sx-x0)**2+(y/sy-y0)**2)/s**2)

class UC_Model_sxy:
    def __init__(self,shape,n_peaks):
        self.shape = shape
        self.n_peaks = n_peaks
        
    def model(self,X,*params):
        
        x,y=X
        p = np.array(params).reshape((self.n_peaks,5))

        return np.sum([self.g(x,y,*pl) for pl in p],axis=0)
    
    def g(self, x,y,A,x0,y0,ssx,ssy):
        
        sy,sx = self.shape
        return A*np.exp(-((x/sx-x0)**2)/ssx**2-((y/sy-y0)**2)/ssy**2)



