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

    def peak_locator(self,min_distance=10):
        self.positions = peak_local_max(self.s.data, min_distance = min_distance)[:,::-1]
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
        print("clustered")
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
        print("positions updated")
        
