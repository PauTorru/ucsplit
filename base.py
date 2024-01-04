import numpy as np
import hyperspy.api as hs
import atomap.api as am
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from ucsplit.refine import *
import scipy.optimize as spo


def polarization_dx(pos):
    A1,A2,A3,A4,B=pos
    center = np.average([A1,A2,A3,A4],axis=0)
    
    x_dir = np.mean(np.array([A2-A1,A4-A3]),axis=0)
    x_dir /=np.linalg.norm(x_dir)
    
    
    
    rel_pos = B-center
    
    return np.dot(rel_pos,x_dir)

def polarization_dy(pos):
    A1,A2,A3,A4,B=pos
    center = np.average([A1,A2,A3,A4],axis=0)
    
    y_dir = np.mean(np.array([A3-A1,A4-A2]),axis=0)
    y_dir/=np.linalg.norm(y_dir)
    
    rel_pos = B-center
    
    return np.dot(rel_pos,y_dir)

def plot_pos_large(pos,im,radius = 3):
    y,x = im.shape
    a = np.zeros([y,x,3])
    
    _im = norm(im)
    
    a[:,:,:] = _im[:,:,np.newaxis]
    
    for p in pos:
        x,y = p.astype("int")
        a[y-radius:y+radius,x-radius:x+radius,:]=np.array([1,0,0])
    
    plt.figure()
    plt.imshow(a)

    
def filter_positions_line(pos,line,remove_where):

    def yl(x):
        return (x-line.x1)*(line.y2-line.y1)/(line.x2-line.x1)+line.y1

    def xl(y):
        return (y-line.y1)*(line.x2-line.x1)/(line.y2-line.y1)+line.x1



    match remove_where:
        case "up":
            def _filter(p):
                x,y = p
                return y>yl(x)

        case "down":
            def _filter(p):
                x,y = p
                return y<yl(x)

        case "left":
            def _filter(p):
                x,y = p
                return x>xl(y)

        case "right":
            def _filter(p):
                x,y = p
                return x<xl(y)

        case default:
            return "remove_where: up | down | left | right"


    w = np.array([_filter(p) for p in pos])

    return pos[w]








def filter_positions(pos,x_max = None,x_min = None,y_max = None,y_min = None):

    if x_max is None:
        x_max = pos[:,0].max()
    if x_min is None:
        x_min = pos[:,0].min()
    if y_max is None:
        y_max = pos[:,1].max()
    if y_min is None:
        y_min = pos[:,1].min()



    fpos = pos.copy()
    for u,d,i in [[x_max,x_min,0],[y_max,y_min,1]]:

        n,_= fpos.shape

        uv,dv = np.ones(n)*u,np.ones(n)*d
        w = np.logical_and(fpos[:,i]>=dv,fpos[:,i]<=uv)

        fpos = fpos[w]


    return fpos





def bob_angle(B1,O,B2):
    """Calculate angle in degrees between atom positions B1-O-B2, with O at vertex.
    
    Parameters:
    -----------
    B1,O,B2 : array

    Returns:
    ---------
    angle : float 
        angle in degrees"""

    _B1=B1-O
    _B2=B2-O
    return (180./np.pi)*np.arccos(np.dot(_B1,_B2)/(np.linalg.norm(_B1)*np.linalg.norm(_B2)))

def bob_angle_from_positions(pos):
    """Calculate angle in degrees between atom positions B1-O-B2, with O at vertex.
    
    Parameters:
    -----------
    pos : array of shape[x,y,[B1x,Ox,B2x],[B1y,Oy,B2y]]

    Returns:
    ---------
    angle : float 
        angle in degrees"""
    B1,O,B2 = pos
    _B1=B1-O
    _B2=B2-O
    return (180./np.pi)*np.arccos(np.dot(_B1,_B2)/(np.linalg.norm(_B1)*np.linalg.norm(_B2)))

def position_image(planes):
    """Generate array with atom coordinates in every pixel.
        

    Parameters:
    -----------
    planes : list 
        list of atom map planes

    Returns:
    ---------
    planeimage : array 
    """

    p_image=np.zeros([len(planes),len(planes[0].x_position),2])
    
    for i,p in enumerate(planes):
        p_image[i,:]=np.vstack([p.x_position,p.y_position]).T
    
    return np.round(p_image,0).astype("int")


def get_uc_signal(image,pimage,left,right,up,down):

    """Generate unitcell signal.

    Parameters:
    -----------
    image : array 
        image to be splitted into unit cells.

    pimage: array
        Position image generated with ucsplit.position_image() from the atomic positions to be used.

    left,right,up,down: int
        Extension of the unit cell in pixels from each atomic position.
        
    Returns:
    ---------
    ucimage : hyperspy.signals.Signal2D
    """

    x,y=pimage.shape[:-1]
    ucs=np.zeros([x,y,up+down,left+right])
    
    for i in range(x):
        for j in range(y):
            px,py=pimage[i,j,...]
            px=int(px)
            py=int(py)
            #print(px,py,i,j)
            ucs[i,j,...]=image[py-up:py+down,px-left:px+right]
    
    return hs.signals.Signal2D(ucs)

def norm(x):
    return (x-x.min())/(x.max()-x.min())

def refine_image_positions_com(pos,image,iters=5,**args):
    _pos=pos.copy().astype("float")
    for p in _pos:
        for _ in range(iters):
            p += refine_atom_position_com(p,image,**args)
    return _pos


def refine_atom_position_com(p,image,radius=5):
    y,x=np.round(p,0).astype("int")
    im=image[x-radius:x+radius,y-radius:y+radius]
    im=norm(im)
    c0,c1=center_of_mass(im)
    return np.array([c1-radius+0.5,c0-radius+0.5])




def _get_uc_signal(image,pimage,roi_info):

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

    x,y=pimage.shape[:-1]
    l,r,u,d = roi_info
    ucs=np.zeros([x,y,d-u,r-l])
    
    for i in range(x):
        for j in range(y):
            px,py=pimage[i,j,...]
            px=int(px)
            py=int(py)
            #print(px,py,i,j)
            ucs[i,j,...]=image[py+u:py+d,px+l:px+r]
    
    return ucs


####################################################################

class UnitCellImage(hs.signals.Signal2D):

    def __init__(self,image=None,uc_centers_matrix=None,data=None,boundx=100,boundy=100,*args,**kwargs):
        
        if not uc_centers_matrix is None:
            self.uc_centers_matrix = uc_centers_matrix.astype("int")
            self._nav_shape=uc_centers_matrix.shape[:2]
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


    def build(self,*args,**kwargs):
        #self.uci=hs.signals.Signal2D(self._get_uc_signal(),*args,**kwargs)
        super().__init__(self._get_uc_signal(),*args,**kwargs)

    def define_uc_roi(self):
        x,y=self.uc_centers_matrix[0,0]
        bx,by=self.bounds
        vsignal=hs.signals.Signal2D(self.original_image[y-by:y+by,x-bx:x+bx])
        vsignal.plot()
        self.roi = hs.roi.RectangularROI(left=bx//2, right=3*bx//2, top=by//2, bottom=3*by//2)
        im_roi = self.roi.interactive(vsignal, color="red")
    
    
    def define_uc_atoms(self):

        self.pos_data = None

        self.pos_data_gui=am.add_atoms_with_gui(self.inav[0,0])


        # put them in eahc unit cell and set pos_data atribute

    def check_pos_data(self):
        if self.pos_data is None:
            self.pos_data=np.zeros(list(self.data.shape[:2])+
                list(np.array(self.pos_data_gui).shape))

            self.pos_data[:,:]=np.array(self.pos_data_gui)[np.newaxis,np.newaxis,:,:]

        if self.markers is None:
            self.uc_add_markers()

    def refine_uc_atoms_com(self,iters=5,**args):
        self.check_pos_data()
        for r in range(self.data.shape[0]):
            for c in range(self.data.shape[1]):
                _pos=self.pos_data.copy()
                self.pos_data[r,c,:,:]=refine_image_positions_com(self.pos_data[r,c],
                    self.data[r,c],iters,**args)
        self.uc_add_markers()


    def refine_uc_atoms_2dgauss(self,sigmas=0.1,fix_sigmas=False):


        xx,yy= np.meshgrid(*[range(i) for i in self.data.shape[-2:]][::-1])
        xy=(xx.ravel(),yy.ravel())
        self.xy=xy
        self._fix_sigmas=fix_sigmas
        if fix_sigmas:
            if isinstance(sigmas,np.ndarray) is False:
                sigmas=np.array(sigmas)
            self._sigmas = sigmas
            self.model = UC_Model_fix_sigma(self.data.shape[-2:],self.pos_data.shape[-2],sigmas).model
            self.gaus_model_params = np.zeros((self.data.shape[0],
            self.data.shape[1],self.pos_data.shape[-2],3))

        else:
            self.model = UC_Model(self.data.shape[-2:],self.pos_data.shape[-2]).model
            self.gaus_model_params = np.zeros((self.data.shape[0],
            self.data.shape[1],self.pos_data.shape[-2],4))

        for r in range(self.data.shape[0]):
            for c in range(self.data.shape[1]):

                im = norm(self.data[r,c])

                if fix_sigmas:
                    params = np.ones((self.pos_data.shape[-2],3))
                    pshape = 3
                else:
                    params = np.ones((self.pos_data.shape[-2],4))
                    params[:,-1] = sigmas
                    pshape = 4


                params[:,0] = im[*self.pos_data[r,c].astype("int").T]
                params[:,1:3] = self.pos_data[r,c]
                params[:,1]/= self.data.shape[-1]
                params[:,2]/= self.data.shape[-2]
                try:
                    res, _ = spo.curve_fit(self.model,
                        self.xy,
                        im.ravel(),
                        p0 = params.ravel(),
                        bounds = (0,1),
                        xtol = 0.001,
                        ftol =1e-3)
                    if c==0:
                        print(r"{}/{}".format(r,self.data.shape[0]))
                except RuntimeError:
                    return r,c


                res = res.reshape((self.pos_data.shape[-2],pshape))
                self.gaus_model_params[r,c]=res


                self.pos_data[r,c,:,0] = res[:,1]*self.data.shape[-1]
                self.pos_data[r,c,:,1] = res[:,2]*self.data.shape[-2]
        self.uc_add_markers()

    def build_2dgaus_model(self):
        mod = np.zeros(self.data.shape)
        for r in range(self.data.shape[0]):
            for c in range(self.data.shape[1]):
                mod[r,c,...]=self.model(self.xy,self.gaus_model_params[r,c,...].ravel()).reshape(self.data.shape[-2:])

        mod = hs.signals.Signal2D(mod)

        plot_compare(self,mod)

        return mod




    def uc_add_markers(self):
        if "Markers" in [i[0] for i in list(self.metadata)]:
            del self.metadata.Markers
        self.markers=[hs.markers.point(self.pos_data[:,:,p,0],self.pos_data[:,:,p,1],color="red") for p in range(self.pos_data.shape[2])]
        self.add_marker(self.markers,permanent=True,plot_marker=False)

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

        x,y=self.uc_centers_matrix.shape[:-1]
        bx,by=self.bounds
        l,r,u,d =[int(i) for i in (self.roi.left-bx,self.roi.right-bx,
                    self.roi.top-by,self.roi.bottom-by)]

        ucs=np.zeros([x,y,d-u,r-l])
        
        for i in range(x):
            for j in range(y):
                px,py=self.uc_centers_matrix[i,j,...].astype("int")
                try:
                    ucs[i,j,...]=self.original_image[py+u:py+d,px+l:px+r]
                    self.uc_centers_matrix[i,j,:]=np.array([px+(r+l)/2,py+(u+d)/2])
                except ValueError:
                    print(px,py)
        
        return ucs


    def get_position_func_image(self,func):

        x,y,p,c=self.pos_data.shape

        return np.array([func(i) for i in self.pos_data.reshape([x*y,p,c])]).reshape([x,y])


    def get_uc_func_image(self,func):
        x,y,p,c=self.pos_data.shape
        x,y,sx,sy=self.data.shape


        return np.array([func(i,j) for i,j in zip( self.pos_data.reshape([x*y,p,c]),
         self.data.reshape([x*y,sx,sy]))]).reshape([x,y])

    def apply_to_ucs(self,func,**kwargs):
        x,y,sx,sy=self.data.shape
        return np.array([func(i,**kwargs) for i in self.data.reshape([x*y,sx,sy])]).reshape(self.data.shape)


    def recenter_uc_to_atom(self,pindex):
        center = np.array(self.data.shape[-2:])/2
        pimage = self.pos_data[:,:,pindex,:] - center[np.newaxis,np.newaxis,:]
        self.uc_centers_matrix+=pimage.astype("int")
        self.define_uc_roi()
        print("Remember to call build() after deining new ROI")
        return


    def create_sublattice_stack(self):

        x,y = self._nav_shape
        self.sublattice_stack=[]

        for i in range(x):
            ss=[]
            for j in range(y):
                ss.append(am.Sublattice(self.pos_data[i,j],self.data[i,j]))

            self.sublattice_stack.append(ss)



    def refine_sublattice_stack(self,use_as_pos_data = True,*args):

        x,y = self._nav_shape

        for i in range(x):
            for j in range(y):
                self.sublattice_stack[i][j].find_nearest_neighbors(nearest_neighbors=self.pos_data.shape[2]-1)
                self.sublattice_stack[i][j].refine_atom_positions_using_2d_gaussian(show_progressbar = False,*args)
                if use_as_pos_data:
                    self.pos_data[i,j] = self.sublattice_stack[i][j].atom_positions
            print(str(i)+"/"+str(x))

    def save(self,*args,**kwargs):
        self.metadata.set_item("UCS",{})
        for k in self._save:
          if k in self.__dict__.keys():
            self.metadata.UCS[k]= self.__dict__[k]

        super().save(*args,**kwargs)

        #del self.metadata.UCS




def load(fname):

    s = hs.load(fname)
    uci = UnitCellImage(uc_centers_matrix = s.metadata.UCS.uc_centers_matrix, data = s.data)

    keys = [*s.metadata.UCS.__dict__.keys()][2:]
    for k in keys:
        uci.__dict__[k] = s.metadata.UCS[k]

    if "_fix_sigmas" in keys:
        if uci._fix_sigmas:
                uci.model = UC_Model_fix_sigma(uci.data.shape[-2:],uci.pos_data.shape[-2],uci._sigmas).model
                uci.gaus_model_params = np.zeros((uci.data.shape[0],
                uci.data.shape[1],uci.pos_data.shape[-2],3))

        else:
                uci.model = UC_Model(uci.data.shape[-2:],uci.pos_data.shape[-2]).model
                uci.gaus_model_params = np.zeros((uci.data.shape[0],
                uci.data.shape[1],uci.pos_data.shape[-2],4))


    if "pos_data" in keys:
        uci.check_pos_data()

    return uci



def plot_line(im):
    line = hs.roi.Line2DROI(0,100,100,200)
    b = hs.signals.Signal2D(im.data)
    b.plot()
    ss = line.interactive(b)
    return line

def plot_average_error(image,axis = 1,name=""):
    plt.figure()
    plt.errorbar(x=range(image.shape[int(abs(axis-1))]),
                 y=image.mean(axis),yerr=image.std(axis),fmt="o",
                ecolor = 'grey', elinewidth = 1, capsize=5)
    plt.tight_layout()
    ax=plt.gca()
    ax.set_ylabel(name)
    plt.gcf().canvas.draw()

    return ax

def plot_compare(s1,s2):
    def follow(obj):
        s2.axes_manager.indices = obj.indices
        s1.axes_manager.indices = obj.indices
    s1.axes_manager.events.indices_changed._connected_all=set()
    s1.axes_manager.events.indices_changed.connect(follow)
    s2.axes_manager.events.indices_changed._connected_all=set()
    s2.axes_manager.events.indices_changed.connect(follow)
    s1.plot()
    s2.plot()
    




