from skimage.feature import peak_local_max
import sklearn
from sklearn.cluster import DBSCAN
import ucsplit.base as uc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


class Uci_Creator:
    def __init__(self,s,markersize=3):
        self.s = s
        self.markersize = markersize
        self._threshold_addpoint=None

    def peak_locator(self,min_distance=10):
        self.positions = peak_local_max(self.s.data, min_distance = min_distance)[:,::-1]
        if self._threshold_addpoint is None:
            self._threshold_addpoint=min_distance//2

        self._threshold_addpoint=min_distance//2
        plt.figure("Atom Positions")
        plt.clf()
        plt.imshow(self.s.data,cmap="gray")
        self.pos_plot = plt.plot(self.positions[:,0],self.positions[:,1],"ro",markersize=self.markersize)

        #uc.plot_pos_large(self.positions,self.s.data)

    def plot(self):
        plt.figure("Atom Positions")
        plt.clf()
        plt.imshow(self.s.data,cmap="gray")
        self.pos_plot = plt.plot(self.positions[:,0],self.positions[:,1],"ro",markersize=self.markersize)

    def construct_zone_axes(self,n_neighbors=9,**kwargs):
        self.positions=np.array(self.positions)
        
        NN = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors)
        neighs = NN.fit(self.positions)
        self.d,self.idx = neighs.kneighbors(self.positions)

        self.relative_positions = self.positions[self.idx]
        self.relative_positions-=self.relative_positions[:,0,:][:,np.newaxis,:]
        self.relative_positions =self.relative_positions[:,1:,:]/np.linalg.norm(self.relative_positions[:,1:,:],axis=2).min()
        self.idx = self.idx[:,1:]
        self.db = DBSCAN(**kwargs).fit(self.relative_positions.reshape((-1,2)))
        self.labels = self.db.labels_.reshape(self.relative_positions.shape[:-1])
        self.centroids=np.array([self.relative_positions.reshape((-1,2))[self.db.labels_==i].mean(0) for i in range(self.db.labels_.max()+1)])
        print("Clustering done.")
        self.plot_zone_axes()

    def select_positions(self,selector_color="limegreen"):
        #uc.plot_pos_large(self.positions,self.s.data)
        self._fig = plt.figure("Atom Positions")
        self._fig.canvas.mpl_connect("close_event",self.onclose)
        plt.clf()
        plt.imshow(self.s.data,cmap="gray")
        self.pos_plot = plt.plot(self.positions[:,0],self.positions[:,1],"ro",markersize=self.markersize)
        self.selector = PolygonSelector(plt.gca(),self.onselect,props={"color" : selector_color,"linewidth":3},useblit=True)

    def plot_zone_axes(self):
        
        plt.clf()
    
        for i in range(len(self.centroids)):
            pts = self.relative_positions.reshape((-1,2))[self.db.labels_==i]
            plt.plot(pts[:,0],-pts[:,1],"o")
            plt.text(self.centroids[i][0],-self.centroids[i][1],str(i),fontsize=20)


    def get_planes(self,id1,id2,i0 = None):
        
        self.taken = np.zeros(self.positions.shape[0],dtype="bool")
        dst = np.linalg.norm(self.positions,axis=1,ord=1)
        if i0 is None:
            i0 = dst.argmin()

        self.taken[i0]=True

        planes = []

        while not self.taken.all():
            #print(i0)
            planes.append(self.create_plane(i0,id1))
            if id2 in self.labels[i0]:
                i0 = self.idx[i0][self.labels[i0]==id2][0]
                self.taken[i0]=True
            elif not self.taken.all():
                print("Could not find neighbour along one of the directions")
                break
            else:
                pass

        return np.array(planes)


    def _old_get_planes(self,id1):
        
        self.taken = np.zeros(self.positions.shape[0],dtype="bool")
        dst = np.linalg.norm(self.positions,axis=1,ord=1)
        i0 = dst.argmin()
        self.taken[i0]=True

        planes = []

        while not self.taken.all():
            #print(i0)
            planes.append(self.create_plane(i0,id1))
            dst[self.taken]=np.inf
            i0 = dst.argmin()

        return np.array(planes)


    def create_plane(self,i0,za):
        i=i0
        self.taken[i0]=True
        plane = [self.positions[i0]]
        while za in self.labels[i]:
            i = self.idx[i][self.labels[i]==za][0]
            self.taken[i]=True
            plane.append(self.positions[i])
            #print(i)
        return np.array(plane)

    def plot_planes(self,planes):
        plt.clf()
        plt.imshow(self.s.data,cmap="gray")
        for p in planes:
            plt.plot(p[:,0],p[:,1])
        


    def onselect(self,verts):
        verts.append(verts[0])
        p = Path(verts)
        self.selected = p.contains_points(self.positions)
        #uc.plot_pos_large([self.positions[selected],self.positions[~selected]],self.s.data,colors=[[1,0,0],[0,1,0]])
        for l in self.pos_plot:
            l.remove()

        self.pos_plot = plt.plot(self.positions[self.selected,0],self.positions[self.selected,1],"go",markersize=self.markersize)
        self.pos_plot.extend(plt.plot(self.positions[~self.selected,0],self.positions[~self.selected,1],"ro",markersize=self.markersize))
        plt.draw()
        self.selector.clear()

    def onclose(self,event):
        self.positions = self.positions[self.selected]
        print("Positions updated.")



     def add_positions(self,):
        print("Positions will be updated when the figure \"Atom positions\" is closed.")
        self._added_pos=[]
        self._fig = plt.figure("Atom Positions")
        self._fig.canvas.mpl_connect("close_event",self.onclose_add_positions)
        plt.clf()
        plt.imshow(self.s.data,cmap="gray")
        self.pos_plot = plt.plot(self.positions[:,0],self.positions[:,1],"ro",markersize=self.markersize)
        self.position_adder = self._fig.canvas.mpl_connect('button_press_event', self.onclick_add_position)

    def remove_positions(self,):
        print("Positions will be updated when the figure \"Atom positions\" is closed.")
        self._added_pos=list(self.positions)
        self._fig = plt.figure("Atom Positions")
        self._fig.canvas.mpl_connect("close_event",self.onclose_remove_positions)
        plt.clf()
        plt.imshow(self.s.data,cmap="gray")
        self.pos_plot = plt.plot(self.positions[:,0],self.positions[:,1],"ro",markersize=self.markersize)
        self.position_remover= self._fig.canvas.mpl_connect('button_press_event', self.onclick_remove_position)

    def onclick_add_position(self,click):
        self.point = [click.xdata,click.ydata]

        if self._is_point_close_to_point_already_added():
            self._remove_point()
        else:
            self._add_point()

    def _is_point_close_to_point_already_added(self):
        if len(self._added_pos)==0:
            return False
        p = np.array(self.point)
        ps = np.array(self._added_pos)
        norm = np.linalg.norm(ps-p,axis=1)
        idx = norm.argmin()
        if norm[idx]<=self._threshold_addpoint:
            self._idx_point_to_remove=idx
            return True
        else:
            return False


    def _remove_point(self):
        self._added_pos.pop(self._idx_point_to_remove)
        line = self.pos_plot.pop(self._idx_point_to_remove+1)
        line.remove()
        plt.draw()

    def _add_point(self):
        self._added_pos.append(self.point)
        self.pos_plot.extend(plt.plot(self.point[0],self.point[1],"go",markersize=self.markersize))
        plt.draw()

    def onclose_add_positions(self,event):
        if len (self._added_pos)>0:
            self.positions = np.concatenate([self.positions,np.array(self._added_pos)],axis=0)
            print("Positions updated")
        else:
            print("No positions added")


    def onclick_remove_position(self,click):
        self.point = [click.xdata,click.ydata]

        if self._is_point_close_to_point_already_added():
            p = self._added_pos.pop(self._idx_point_to_remove)
            self.pos_plot.extend(plt.plot(p[0],p[1],"bo",markersize=self.markersize))
            plt.draw()
        else:
            pass

    def onclose_remove_positions(self,event):
        if len (self._added_pos)>0:
            self.positions = np.array(self._added_pos)
            print("Positions updated")
        else:
            print("No positions removed")
        
class Add_Delete_Positions:
    def __init__(self,image,initial_positions,markersize=1):
        self.image = image
        self.init_pos = initial_positions.copy()
        self.final_pos = initial_positions.copy()
        self.idx_pos_to_remove =[]
        self.pos_to_add = []
        self.markersize=markersize

        self._fig = plt.figure("Add/Remove positions")
        plt.clf()
        plt.imshow(self.image,cmap="gray")
        self.pos_plot = plt.plot(self.init_pos[0],self.init_pos[1],"ro",markersize=self.markersize)
        self._fig.canvas.mpl_connect("close_event",self.onclose_AddDelete)
        self.position_AddDelete= self._fig.canvas.mpl_connect('button_press_event', self.onclick_AddDelete)

    def onclose_AddDelete(self)
        for i in sorted(self.idx_pos_to_remove)[::-1]:
            _=self.final_pos.pop(i)
        self.final_pos = np.concatenate(self.final_pos,np.array(self.pos_to_add))

    def onclick_AddDelete(self):
        pass
        #four cases : 
        # 1 close to atom already there, that is initial but marked for removal
        # 2 close to atom already there, that is initial and not marked for removal
        # 3 close to atom already there that is newly added
        # 4 close to nothin

