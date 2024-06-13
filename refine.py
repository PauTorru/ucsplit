
import numpy as np

class UC_Model_full_fixed_atoms:
    def _init_(self,shape,n_peaks,fixed,fixed_params):

        if sorted(fixed)!=fixed or not isinstance(fixed,list):
            raise TypeError("Sorted list required")

        self.shape = shape
        self.n_peaks = n_peaks
        self.fixed=fixed
        self.fixed_params=fixed_params

    def model(self,X,*params):
        
        x,y=X
        p = np.array(params).reshape((self.n_peaks,5))
        for i,pi in zip(self.fixed,self.fixed_params):
            p = np.insert(p, i, pi, axis=0)

        return np.sum([self.g(x,y,*pl) for pl in p],axis=0)
    
    def g(self, x,y,A,x0,y0,ssx,ssy):
        
        sy,sx = self.shape
        return A*np.exp(-((x/sx-x0)**2)/ssx**2-((y/sy-y0)**2)/ssy**2)

    def dg(self,x,y,A,x0,y0,ssx,ssy):

        sy,sx = self.shape


        ex = -((x/sx-x0)**2)/ssx**2
        ey = -((y/sy-y0)**2)/ssy**2
        ee =np.exp(ex+ey)
        T = A*ee
        dA = ee

        dx0 = T*2*(x/sx-x0)/(ssx**2)
        dy0 = T*2*(y/sy-y0)/(ssy**2)
        dsx = -2*ex*T/(ssx**3)
        dsy = -2*ex*T/(ssx**3)
        return [dA,dx0,dy0,dsx,dsy]


    def jacobian(self,X,*params):
        
        x,y=X
        p = np.array(params).reshape((self.n_peaks,5))
        for i,pi in zip(self.fixed,self.fixed_params):
            p = np.insert(p, i, pi, axis=0)

        jac = np.array([self.dg(x,y,*pl) for pl in p])
        return jac.reshape([-1,jac.shape[-1]]).T






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

    def dg(self, x,y,A,x0,y0,s):

        sy,sx = self.shape


        e =-((x/sx-x0)**2+(y/sy-y0)**2)
        ee =np.exp(e/s**2)
        T = A*ee
        dA = ee
        dx0 = T*2*(x/sx-x0)/(s**2)
        dy0 = T*2*(y/sy-y0)/(s**2)
        ds = -2*e*T/(s**3)
        return [dA,dx0,dy0,ds]


    def jacobian(self,X,*params):
        x,y=X
        p = np.array(params).reshape((self.n_peaks,4))
        jac = np.array([self.dg(x,y,*pl) for pl in p])
        return jac.reshape([-1,jac.shape[-1]]).T


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


    def dg(self, x,y,A,x0,y0,s):

        sy,sx = self.shape


        e =-((x/sx-x0)**2+(y/sy-y0)**2)
        ee =np.exp(e/s**2)
        T = A*ee
        dA = ee
        dx0 = T*2*(x/sx-x0)/(s**2)
        dy0 = T*2*(y/sy-y0)/(s**2)
        return [dA,dx0,dy0]


    def jacobian(self,X,*params):
        x,y=X
        p = np.array(params).reshape((self.n_peaks,3))
        p = np.hstack([p,self.sigmas[:,np.newaxis]])

        jac = np.array([self.dg(x,y,*pl) for pl in p])
        return jac.reshape([-1,jac.shape[-1]]).T


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

    def dg(self,x,y,A,x0,y0,ssx,ssy):

        sy,sx = self.shape


        ex = -((x/sx-x0)**2)/ssx**2
        ey = -((y/sy-y0)**2)/ssy**2
        ee =np.exp(ex+ey)
        T = A*ee
        dA = ee

        dx0 = T*2*(x/sx-x0)/(ssx**2)
        dy0 = T*2*(y/sy-y0)/(ssy**2)
        dsx = -2*ex*T/ssx
        dsy = -2*ey*T/ssy
        return [dA,dx0,dy0,dsx,dsy]


    def jacobian(self,X,*params):
        x,y=X
        p = np.array(params).reshape((self.n_peaks,5))
        jac = np.array([self.dg(x,y,*pl) for pl in p])
        return jac.reshape([-1,jac.shape[-1]]).T




