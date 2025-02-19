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

def fix_planes(p):
    
    l = max([len(j.atom_list) for j in p])
    print(l)
    ls = [len(i.atom_list) for i in p]
    good = [i==l for i in ls]
    orientation = np.array(p[0].zone_vector).argmax()

    while not all(good):
        i1 = good.index(False)
        i2 = good[i1+1:].index(False)+i1+1

        if len(p[i1].atom_list)+len(p[i2].atom_list)>l:
            print("not fixable")
            return

        if orientation==0:
            keep =np.array([min(j.get_x_position_list()) for j in[p[i1],p[i2]]]).argmin()
        else:
            keep =np.array([min(j.get_y_position_list()) for j in[p[i1],p[i2]]]).argmin()

        if keep == 0:
            p[i1].atom_list+=p[i2].atom_list
            p.pop(i2)
        if keep ==1:
            p[i2].atom_list+=p[i1].atom_list
            p.pop(i1)
            
        l = max([len(j.atom_list) for j in p])
        ls = [len(i.atom_list) for i in p]
        good = [i==l for i in ls]        

    return p

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
        x,y = np.round(p,0).astype("int")
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





def bob_angle(pos):
    """Calculate angle in degrees between atom positions B1-O-B2, with O at vertex.
    
    Parameters:
    -----------
    B1,O,B2 : array

    Returns:
    ---------
    angle : float 
        angle in degrees"""
    B1,O1,B2,O2,B3=pos

    _B1=B1-O1
    _B2=B2-O1
    angle1 = (180./np.pi)*np.arccos(np.dot(_B1,_B2)/(np.linalg.norm(_B1)*np.linalg.norm(_B2)))
    
    _B1=B2-O2
    _B2=B3-O2
    angle2 = (180./np.pi)*np.arccos(np.dot(_B1,_B2)/(np.linalg.norm(_B1)*np.linalg.norm(_B2)))

    return np.average([angle1,angle2])


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
    
    return p_image


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
            #p = refine_atom_position_com(p,image,**args)
    return _pos


def refine_atom_position_com(p,image,radius=5):
    x,y=p #y,x=np.round(p,0).astype("int")
    rr,cc = skimage.draw.disk((y,x),radius,shape = image.shape)
    im = np.zeros_like(image)
    im[rr,cc] = norm(image)[rr,cc]

    return np.array(center_of_mass(im))[::-1]-p




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
        if self.original_image.ndim ==2:
            vsignal=hs.signals.Signal2D(self.original_image[y-by:y+by,x-bx:x+bx])
            vsignal.plot()
        if self.original_image.ndim ==3:
            vsignal=hs.signals.Signal2D(self.original_image[y-by:y+by,x-bx:x+bx].sum(-1))
            vsignal.plot()

        self.roi = hs.roi.RectangularROI(left=bx//2, right=3*bx//2, top=by//2, bottom=3*by//2)
        im_roi = self.roi.interactive(vsignal, color="red")
    
    
    def define_uc_atoms(self):

        self.pos_data = None
        self.markers = None
        self.pos_data_gui=am.add_atoms_with_gui(self.inav[0,0],distance_threshold=1)


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
            print(r,"/",self.data.shape[0])
            for c in range(self.data.shape[1]):
                _pos=self.pos_data.copy()
                self.pos_data[r,c,:,:]=refine_image_positions_com(self.pos_data[r,c],
                    self.data[r,c],iters,**args)
        self.uc_add_markers()

    def refine_uc_atoms_2dgauss_smart(self):
        pass
        
    def refine_uc_atoms_2dgauss(self,sigmas=0.1,model="default",bounds = (0,1),xtol=0.001,ftol=1e-3,use_jacobian=True,loss="linear"):
        r""" mode: "default", "fix_sigmas", "full2d" """


        xx,yy= np.meshgrid(*[range(i) for i in self.data.shape[-2:]][::-1])
        xy=(xx.ravel(),yy.ravel())
        self.xy=xy
        if model =="fix_sigmas":
            if isinstance(sigmas,np.ndarray) is False:
                sigmas=np.array(sigmas)
            self._sigmas = sigmas
            self.model = UC_Model_fix_sigma(self.data.shape[-2:],self.pos_data.shape[-2],sigmas).model
            self.jacobian = UC_Model_fix_sigma(self.data.shape[-2:],self.pos_data.shape[-2],sigmas).jacobian
            self.gaus_model_params = np.zeros((self.data.shape[0],
            self.data.shape[1],self.pos_data.shape[-2],3))
            pshape=3
            init_params = np.ones((self.pos_data.shape[-2],pshape))

        elif model =="default":
            self.model = UC_Model(self.data.shape[-2:],self.pos_data.shape[-2]).model
            self.jacobian = UC_Model(self.data.shape[-2:],self.pos_data.shape[-2]).jacobian
            self.gaus_model_params = np.zeros((self.data.shape[0],
            self.data.shape[1],self.pos_data.shape[-2],4))
            pshape=4
            init_params = np.ones((self.pos_data.shape[-2],pshape))
            init_params[:,-1] = sigmas

        elif model=="full2d":
            self.model = UC_Model_sxy(self.data.shape[-2:],self.pos_data.shape[-2]).model
            self.jacobian = UC_Model_sxy(self.data.shape[-2:],self.pos_data.shape[-2]).jacobian
            self.gaus_model_params = np.zeros((self.data.shape[0],
            self.data.shape[1],self.pos_data.shape[-2],5))
            pshape=5
            init_params = np.ones((self.pos_data.shape[-2],pshape))
            init_params[:,-1]=sigmas
            init_params[:,-2]=sigmas

        elif model=="full2d_rotation":
            self.model = UC_Model_rotation(self.data.shape[-2:],self.pos_data.shape[-2]).model
            self.jacobian = UC_Model_rotation(self.data.shape[-2:],self.pos_data.shape[-2]).jacobian
            self.gaus_model_params = np.zeros((self.data.shape[0],
            self.data.shape[1],self.pos_data.shape[-2],6))
            pshape=6
            init_params = np.ones((self.pos_data.shape[-2],pshape))
            init_params[:,-2]=sigmas
            init_params[:,-3]=sigmas
            init_params[:,-1]=sigmas
        else:
            return "Model Not available"





        if not use_jacobian:
            self.jacobian = None




        for r in range(self.data.shape[0]):
            for c in range(self.data.shape[1]):

                im = norm(self.data[r,c])

                idxs = self.pos_data[r,c].astype("int")[:,::-1].T
                init_params[:,0] = im[[i for i in idxs[0]],[i for i in idxs[1] ]]
                #init_params[:,0] = im[*self.pos_data[r,c].astype("int")[:,::-1].T]
                init_params[:,1:3] = self.pos_data[r,c]
                init_params[:,1]/= self.data.shape[-1]
                init_params[:,2]/= self.data.shape[-2]
                try:
                    res, _ = spo.curve_fit(self.model,
                        self.xy,
                        im.ravel(),
                        p0 = init_params.ravel(),
                        bounds = bounds,
                        xtol = xtol,
                        ftol = ftol,
                        method="trf",jac = self.jacobian,loss=loss)
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

        self.markers=[]
        for atom in range(self.pos_data.shape[2]):
            offsets = np.empty(self.axes_manager.navigation_shape, dtype=object)
            marker_pos = self.pos_data[:,:,atom,:].transpose([1,0,2])
            for i in np.ndindex(self.axes_manager.navigation_shape):
                offsets[i] = [marker_pos[i],]

            marker = hs.plot.markers.Points(offsets=offsets,color="red",sizes=5)
            self.markers.append(marker)

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

    def flatten_positions(self,radius=5):
        """Returns positions of UCI, accounting for overlaps if the positions are closer than \" radius\" ."""
        pd = self.pos_data
        centers = self.uc_centers_matrix
        shift = np.array(self.data.shape[-2:])/2
        
        flat_positions = []
        pos_data_flat_id = np.zeros(pd.shape[:-1])
        k=0
        nrows,ncols,natoms,ncoords = pd.shape
        for r in range(nrows):
            for c in range(ncols):
                for atom in range(natoms):
                    p = centers[r,c]
                    a_pos = p-shift+pd[r,c,atom]
                    flat_positions.append(a_pos)
                    pos_data_flat_id[r,c,atom]=k
                    k+=1

        flat_pos = np.array(flat_positions)
        NN = sklearn.neighbors.NearestNeighbors(n_neighbors=natoms)
        neighs = NN.fit(flat_pos)
        d,idx = neighs.kneighbors(flat_pos)
        filtered = [tuple(sorted(i[j]))for i,j in zip(idx,d<radius)]
        uniques = list(set(filtered))
        
        actual_pos_data_id =[]
        for idd in pos_data_flat_id.ravel():
            actual_pos_data_id.append(np.array([idd in i for i in uniques]).argmax())
        actual_pos_data_id = np.array(actual_pos_data_id).reshape(pd.shape[:-1])

        self.pos_data_flat_id=actual_pos_data_id
        self.flat_uniques = np.array([flat_pos[list(i)].mean(0) for i in uniques])


        return self.flat_uniques

    def calibrate_with_distance_between_equivalents(self,n_atom,direction,range,physical_distance,unit="nm"):
    """Calibrate pixel size using the distance between each equivalent atom in range.

    Parameters:
    -----------
    n_atom : int 
        Index of atom for which the distance to its neighbouring equivalent will be calculated

    direction: int
        0 == vertical direction (distance between atom and the equivalent in the next row)
        1 == horizontal direction (distance between atom and the equivalent in the nex column)

    range: np.s_
        Slicer for which unitcells should be used. E.g. np.s_[:10,:] to calibrate using only the first ten rows of unit cells.

    physical_distance: float
        Value of the expected physical distance between equivalent atoms.

    unit: str
        units of the specified physical distance (e.g. "nm","um",...).
        
    Returns:
    ---------
    pixel_size: float
    
    """
    if not hasattr(self,"flat_uniques"):
        self.flatten_positions()

    actual_positions = self.flat_uniques[self.pos_data_flat_id]

    reference_atom_positions= actual_positions[:,:,n_atom,:]
    
    position_to_use = reference_atom_positions[range]

    if direction == 0: 
        mean_pixel_distance = np.linalg.norm(position_to_use[1:]-position_to_use[:-1],axis=-1).mean((0,1))
    elif direction ==1:
        mean_pixel_distance = np.linalg.norm(position_to_use[:,1:]-position_to_use[:,:-1],axis=-1).mean((0,1))
    else:
        raise Exception("Direction has to be 0 or 1")

    pixel_size = physical_distance/mean_pixel_distance
    self.uci_calibrated_scale_unit="nm"
    self.uci_calibrated_scale = pixel_size
    return pixel_size

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
        else:
                uci.model = UC_Model(uci.data.shape[-2:],uci.pos_data.shape[-2]).model


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


def plot_polarization(uci,dxi=None,dyi=None,k=1,scale=20,color="yellow",head_width=10,**kwargs):

    #plt.figure()
    plt.imshow(uci.original_image,cmap="gray")

    if dxi is None:
        dxi = uci.get_position_func_image(polarization_dx)
    if dyi is None:
        dyi = uci.get_position_func_image(polarization_dy)
    pos = uci.uc_centers_matrix
    nx,ny = dxi.shape

    red_px,red_py,red_cx,red_cy = [skimage.measure.block_reduce(i[:nx//k*k,:ny//k*k],
     k, np.mean).flatten() for i in [dxi,dyi,pos[:,:,0],pos[:,:,1]]]

    for px,py,cx,cy in zip(*[i for i in [red_px,red_py,red_cx,red_cy]]):
        plt.arrow(cx,cy,-scale*px,-scale*py,color="yellow",head_width = head_width,**kwargs)

    plt.tight_layout()

    return dxi,dyi

def get_polarization_mod_angle(uci,pxi=None,pyi=None):
    if pxi is None:
        pxi = uci.get_position_func_image(polarization_dx)
    if pyi is None:
        pyi = uci.get_position_func_image(polarization_dy)
    pai = -np.arctan2(pyi,pxi)*180/np.pi+180
    pmi = np.linalg.norm(np.dstack([pxi,pyi]),axis=-1)
    return pmi,pai


def plot_polarization_mod_angle(uci,pxi=None,pyi=None,**params):
    pmi,pai=get_polarization_mod_angle(uci,pxi,pyi)


    fig = plt.gcf()
    fig.clf()

    fig,[ax1,ax2] = plt.subplots(2,1,figsize=(12,4), num = fig.number)




    ax1.imshow(pai,vmin=0,vmax=360,cmap = "twilight")
    title="Polarization Angle"
    add_uci_scale_bar(ax1,uci,**params)
    #cb = plt.colorbar()
    #cb.set_label("degrees")


    #plt.sca(ax2)

    pmi_im = ax2.imshow(pmi)
    title="Polarization modulus"
    add_uci_scale_bar(ax2,uci,**params)
    cb = plt.colorbar(pmi_im,ax=ax2,shrink = 0.8)
    cb.set_label("px")

    bs = ax1.get_position()
    #cb_bs = [bs.xmax,bs.ymax-2*bs.height/4,bs.width/5,bs.height/5]
    cb_bs = [bs.xmax+bs.width/12,bs.ymax-bs.height/2,bs.width/6,bs.height/6]

    display_axes = fig.add_axes(cb_bs, projection='polar')
    """display_axes._direction = 2*np.pi ## This is a nasty hack - using the hidden field to 
                                      ## multiply the values such that 1 become 2*pi
                                      ## this field is supposed to take values 1 or -1 only!!"""

    norm = mpl.colors.Normalize(0,2*np.pi)

    # Plot the colorbar onto the polar axis
    # note - use orientation horizontal so that the gradient goes around
    # the wheel rather than centre out
    quant_steps = 2056
    cb = mpl.colorbar.ColorbarBase(display_axes, cmap=cm.get_cmap('twilight',quant_steps),
                                       norm=norm,
                                       orientation='horizontal')

    # aesthetics - get rid of border and axis labels                                   
    cb.outline.set_visible(False) 

    cb.set_ticks([0,np.pi/2,np.pi,3*np.pi/2])

    display_axes.set_rlim([-1,1])

    plt.tight_layout()

    return pai,pmi





def add_uci_scale_bar(ax,uci,unit_size = 20,unit_name="nm",fontsize = 18, *params):
    ax.set_xticks([])
    ax.set_yticks([])
    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(ax.transData,
                           unit_size/(uci.data.shape[-1]*uci.original_scale), str(unit_size)+unit_name, 'lower right', 
                           pad=0.1,
                           color='white',
                           frameon=False,
                           size_vertical=1,
                           fontproperties=fontprops)

    ax.add_artist(scalebar)


def add_scale_bar(ax,s,unit_size = 20,unit_name="nm",fontsize = 18,pad=0.1,size_vertical=2, *params,):
    ax.set_xticks([])
    ax.set_yticks([])
    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(ax.transData,
                           unit_size/(s.axes_manager[0].scale), str(unit_size)+unit_name, 'lower right', 
                           pad=pad,
                           color='white',
                           frameon=False,
                           size_vertical=2,
                           fontproperties=fontprops)

    ax.add_artist(scalebar)

def fix_old_save(fname):
    with h5py.File(fname,"r+") as f:
        del f["Experiments"]['__unnamed__']["metadata"]["Markers"]
        f.close()
        return

def plot_planes(s,planes):
    plt.clf()
    plt.imshow(s.data,cmap="gray")

    for p in planes:
        plt.plot(p.x_position,p.y_position,"-")

def uci_image2image(uci_image,uci):
    pass







    




