import numpy as np
import hyperspy.api as hs
import atomap.api as am
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from ucsplit.refine import *
import scipy.optimize as spo
import skimage
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib import cm
import h5py
import sklearn
import base as b





class FreeUnitCellImage(hs.signals.Signal2D):



def __init__(self,image=None,uc_centers_list=None,data=None,boundx=100,boundy=100,*args,**kwargs):
        
        if not uc_centers_matrix is None:
            self.uc_centers_list = uc_centers_list

       

        if not image is None:
            self.original_scale = image.axes_manager[0].scale
            self.original_image = image.data
        
        self.bounds = (boundx,boundy)
        self.pos_data = None
        self.markers = None

        self._save = ['uc_centers_matrix',
      'original_image', 'bounds','original_scale','_sigmas',
      'pos_data', 'xy', '_fix_sigmas', 'gaus_model_params',"_save"]
        if data is None:
            self.uc_roi = self.define_uc_roi()

        else:
            super().__init__(data,*args,**kwargs)#self.uci = hs.signals.Signal2D(uc_data)




    def define_uc_roi(self):
        x,y=self.uc_centers_matrix[0][0]
        bx,by=self.bounds
        if self.original_image.ndim ==2:
            vsignal=hs.signals.Signal2D(self.original_image[y-by:y+by,x-bx:x+bx])
            vsignal.plot()
        if self.original_image.ndim ==3:
            vsignal=hs.signals.Signal2D(self.original_image[y-by:y+by,x-bx:x+bx].sum(-1))
            vsignal.plot()

        self.roi = hs.roi.RectangularROI(left=bx//2, right=3*bx//2, top=by//2, bottom=3*by//2)
        im_roi = self.roi.interactive(vsignal, color="red")
    

    def build(self,*args,**kwargs):
        super().__init__(self._get_uc_signal(),*args,**kwargs)




    def _get_uc_signal(self):

        """Generate unitcell signal.

        Parameters:
        -----------
        image : array 
            image to be splitted into unit cells.

        pimage: array
            Position image generated with ucsplit.position_image() from the atomic positions to be used.

        roi: hyperspy roi
            Extension of the unit cell for each atomic position.
            
        Returns:
        ---------
        ucimage : np.array
        """

        x,y=self.uc_centers_matrix.shape[-1]

        bx,by=self.bounds
        l,r,u,d =[int(i) for i in (self.roi.left-bx,self.roi.right-bx,
                    self.roi.top-by,self.roi.bottom-by)]

        if self.original_image.ndim==2:
            ucs=np.zeros([x,y,d-u,r-l])
        if self.original_image.ndim==3:
            ucs=np.zeros([x,y,d-u,r-l,self.original_image.shape[-1]])
        
        for i in range(x):
            for j in range(y):
                px,py=np.round(self.uc_centers_matrix[i,j,...],0).astype("int")
                try:
                    ucs[i,j,...]=self.original_image[py+u:py+d,px+l:px+r]
                    self.uc_centers_matrix[i,j,:]=np.array([px+(r+l)/2,py+(u+d)/2])
                except ValueError:
                    print(px,py)
        
        return ucs