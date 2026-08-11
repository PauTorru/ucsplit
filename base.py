import numpy as np
import hyperspy.api as hs
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import scipy.optimize as spo
import skimage
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib import cm
import h5py
import sklearn
from .creator import Add_Delete_Positions
import os
import tempfile
import zipfile
import pickle
from .refine import RefinePositions


class UnitCellImage(hs.signals.Signal2D, RefinePositions):
    def __init__(
        self,
        image=None,
        uc_centers_matrix=None,
        data=None,
        boundx=100,
        boundy=100,
        *args,
        **kwargs,
    ):

        if not uc_centers_matrix is None:
            self.uc_centers_matrix = uc_centers_matrix.astype("int")
            self.nav_shape = uc_centers_matrix.shape[:2]
        if not image is None:
            self.original_scale = image.axes_manager[0].scale
            self.original_image = image.data

        self.bounds = (boundx, boundy)
        self.markers = None

        if data is None:
            self.uc_roi = self.define_uc_roi()
            super().__init__(np.eye(2), *args, **kwargs)  # place holder initialization
            fig = plt.gcf()
            fig.canvas.mpl_connect("close_event", self.init_onclose)

        else:
            super().__init__(data, *args, **kwargs)

    def init_onclose(self, event):
        self.build()

    def build(self, *args, **kwargs):
        super().__init__(self._get_uc_signal(), *args, **kwargs)

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

        x, y = self.uc_centers_matrix.shape[:-1]
        bx, by = self.bounds
        l, r, u, d = [
            int(i)
            for i in (
                self.roi.left - bx,
                self.roi.right - bx,
                self.roi.top - by,
                self.roi.bottom - by,
            )
        ]
        self.roi_bounds = [
            i for i in (self.roi.left, self.roi.right, self.roi.top, self.roi.bottom)
        ]
        if self.original_image.ndim == 2:
            ucs = np.zeros([x, y, d - u, r - l])
        if self.original_image.ndim == 3:
            ucs = np.zeros([x, y, d - u, r - l, self.original_image.shape[-1]])

        self.uc_slicers = {}
        for i in range(x):
            for j in range(y):
                px, py = np.round(self.uc_centers_matrix[i, j, ...], 0).astype("int")
                try:
                    ucs[i, j, ...] = self.original_image[
                        py + u : py + d, px + l : px + r
                    ]
                    self.uc_centers_matrix[i, j, :] = np.array(
                        [px + (r + l) / 2, py + (u + d) / 2]
                    )
                    self.uc_slicers[(i, j)] = np.s_[py + u : py + d, px + l : px + r]
                except ValueError:
                    print(px, py)
        return ucs

    def define_uc_roi(self):
        x, y = self.uc_centers_matrix[0, 0]
        bx, by = self.bounds
        if self.original_image.ndim == 2:
            vsignal = hs.signals.Signal2D(
                self.original_image[y - by : y + by, x - bx : x + bx]
            )
            vsignal.plot()
        if self.original_image.ndim == 3:
            vsignal = hs.signals.Signal2D(
                self.original_image[y - by : y + by, x - bx : x + bx].sum(-1)
            )
            vsignal.plot()

        self.roi = hs.roi.RectangularROI(
            left=bx // 2, right=3 * bx // 2, top=by // 2, bottom=3 * by // 2
        )
        im_roi = self.roi.interactive(vsignal, color="red")

    def define_uc_atoms(self, markersize=3, proximity_tol=1):

        self.pos_data = None
        self.markers = None
        self._uc_atoms_gui = Add_Delete_Positions(
            self.data.mean((0, 1)),
            np.array([]),
            markersize,
            proximity_tol=proximity_tol,
            onclose_callback=self._uc_atoms_onclose_callback,
        )

    def _uc_atoms_onclose_callback(self, _):
        self.pos_data_gui = self._uc_atoms_gui.final_positions
        self.check_pos_data()

    def check_pos_data(self):
        if self.pos_data is None:
            self.pos_data = np.zeros(
                list(self.data.shape[:2]) + list(np.array(self.pos_data_gui).shape)
            )

            self.pos_data[:, :] = np.array(self.pos_data_gui)[
                np.newaxis, np.newaxis, :, :
            ]

        if self.markers is None:
            self.uc_add_markers()

    def plot_uc_atoms(self, fontsize=15):
        fig = plt.figure("Unit Cell Positions")
        plt.imshow(self.mean((0, 1)))
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        for ii, p in enumerate(self.pos_data_gui):
            plt.plot(p[0], p[1], "ro")
            plt.text(p[0], p[1], str(ii), fontsize=fontsize)

    def uc_add_markers(self):
        if "Markers" in [i[0] for i in list(self.metadata)]:
            del self.metadata.Markers

        self.markers = []
        for atom in range(self.pos_data.shape[2]):
            offsets = np.empty(self.axes_manager.navigation_shape, dtype=object)
            marker_pos = self.pos_data[:, :, atom, :].transpose([1, 0, 2])
            for i in np.ndindex(self.axes_manager.navigation_shape):
                offsets[i] = [
                    marker_pos[i],
                ]

            marker = hs.plot.markers.Points(offsets=offsets, color="red", sizes=5)
            self.markers.append(marker)

        self.add_marker(self.markers, permanent=True, plot_marker=False)

    def map_uc_property(self, function):
        """
        Maps a property of the atomic columns positions in a unit-cell for all unit cells:

        Parameters:
        -----------

        function: function
                f: atom_positions -> scalar

        Returns:
        --------

        np.array of shape (n_unitcells_x,n_unitcells_y)
        """
        x, y, p, c = self.pos_data.shape

        return np.array(
            [function(i) for i in self.pos_data.reshape([x * y, p, c])]
        ).reshape([x, y])

    def recenter_uc_to_atom(self, pindex):
        center = np.array(self.data.shape[-2:]) / 2
        pimage = self.pos_data[:, :, pindex, :] - center[np.newaxis, np.newaxis, :]
        self.uc_centers_matrix += pimage.astype("int")
        self.define_uc_roi()
        fig = plt.gcf()
        fig.canvas.mpl_connect("close_event", self.init_onclose)
        return

    def flatten_positions(self, radius=5):
        """Returns positions of UCI, accounting for overlaps if the positions are closer than \" radius\" ."""
        pd = self.pos_data
        centers = self.uc_centers_matrix
        shift = np.array(self.data.shape[-2:]) / 2

        flat_positions = []
        pos_data_flat_id = np.zeros(pd.shape[:-1])
        k = 0
        nrows, ncols, natoms, ncoords = pd.shape
        for r in range(nrows):
            for c in range(ncols):
                for atom in range(natoms):
                    p = centers[r, c]
                    a_pos = p - shift + pd[r, c, atom]
                    flat_positions.append(a_pos)
                    pos_data_flat_id[r, c, atom] = k
                    k += 1

        flat_pos = np.array(flat_positions)
        NN = sklearn.neighbors.NearestNeighbors(n_neighbors=natoms)
        neighs = NN.fit(flat_pos)
        d, idx = neighs.kneighbors(flat_pos)
        filtered = [tuple(sorted(i[j])) for i, j in zip(idx, d < radius)]
        uniques = list(set(filtered))

        actual_pos_data_id = []
        for idd in pos_data_flat_id.ravel():
            actual_pos_data_id.append(np.array([idd in i for i in uniques]).argmax())
        actual_pos_data_id = np.array(actual_pos_data_id).reshape(pd.shape[:-1])

        self.pos_data_flat_id = actual_pos_data_id
        self.flat_uniques = np.array([flat_pos[list(i)].mean(0) for i in uniques])

        return self.flat_uniques

    def calibrate_with_distance_between_equivalents(
        self, n_atom, direction, slice_range, physical_distance, unit="nm"
    ):
        """Calibrate pixel size using the distance between each equivalent atom in range.

        Parameters:
        -----------
        n_atom : int
                Index of atom for which the distance to its neighbouring equivalent will be calculated

        direction: int
                0 == vertical direction (distance between atom and the equivalent in the next row)
                1 == horizontal direction (distance between atom and the equivalent in the nex column)

        slice_range: np.s_
                Slicer for which unitcells should be used. E.g. np.s_[:10,:] to calibrate using only the first ten rows of unit cells.

        physical_distance: float
                Value of the expected physical distance between equivalent atoms.

        unit: str
                units of the specified physical distance (e.g. "nm","um",...).

        Returns:
        ---------
        pixel_size: float

        """
        if not hasattr(self, "flat_uniques"):
            self.flatten_positions()

        actual_positions = self.flat_uniques[self.pos_data_flat_id]

        reference_atom_positions = actual_positions[:, :, n_atom, :]

        position_to_use = reference_atom_positions[slice_range]

        if direction == 0:
            mean_pixel_distance = np.linalg.norm(
                position_to_use[1:] - position_to_use[:-1], axis=-1
            ).mean((0, 1))
        elif direction == 1:
            mean_pixel_distance = np.linalg.norm(
                position_to_use[:, 1:] - position_to_use[:, :-1], axis=-1
            ).mean((0, 1))
        else:
            raise Exception("Direction has to be 0 or 1")

        pixel_size = physical_distance / mean_pixel_distance
        self.uci_calibrated_scale_unit = unit
        self.uci_calibrated_scale = pixel_size
        return pixel_size

    def reflatten_to_image(self, array=None, default_fill=0.0):
        """Reconstructs the full original 2D image or 3D spectrum image from unit cell patches

        stored in `self.data` using weighted accumulation (running average).

        Parameters:
        -----------
        array : np.array
                array to be flattened. if None self.data is used.

        default_fill : float
                Background value for canvas pixels not covered by any unit cell patch.

        Returns:
        --------
        reconstructed : np.ndarray
                Reconstructed array matching `self.original_image.shape`.
                Shape is (height, width) for 2D, or (height, width, channels) for 3D.
        """
        x, y = self.uc_centers_matrix.shape[:2]
        bx, by = self.bounds

        # Compute relative bounding box parameters from ROI exactly like in _get_uc_signal
        l, r, u, d = [int(i) for i in self.roi_bounds]

        out_shape = self.original_image.shape
        is_3d = self.original_image.ndim == 3

        # Prepare accumulator array matching full data dimensions & 2D count map
        accumulator = np.zeros(out_shape, dtype=np.float64)
        counts = np.zeros(out_shape[:2], dtype=np.float64)
        if array is not None:
            data = array
        else:
            data = self.data

        for i in range(x):
            for j in range(y):
                px, py = np.round(self.uc_centers_matrix[i, j, ...], 0).astype("int")

                # Canvas spatial bounds with border protection
                y_start, y_end = max(0, py + u), min(out_shape[0], py + d)
                x_start, x_end = max(0, px + l), min(out_shape[1], px + r)

                # Local patch indices matching clamped canvas slice
                patch_y_start = y_start - (py + u)
                patch_y_end = patch_y_start + (y_end - y_start)
                patch_x_start = x_start - (px + l)
                patch_x_end = patch_x_start + (x_end - x_start)

                if is_3d:
                    accumulator[y_start:y_end, x_start:x_end, :] += data[
                        i,
                        j,
                        patch_y_start:patch_y_end,
                        patch_x_start:patch_x_end,
                        :,
                    ]
                else:
                    accumulator[y_start:y_end, x_start:x_end] += data[
                        i,
                        j,
                        patch_y_start:patch_y_end,
                        patch_x_start:patch_x_end,
                    ]

                counts[y_start:y_end, x_start:x_end] += 1.0

        # Construct final array
        reconstructed = np.full(out_shape, default_fill, dtype=data.dtype)
        mask = counts > 0

        if is_3d:
            # Broadcast mask over channel dimension
            reconstructed[mask, :] = accumulator[mask, :] / counts[mask, np.newaxis]
        else:
            reconstructed[mask] = accumulator[mask] / counts[mask]

        return reconstructed

    def save(self, filename, *args, **kwargs):
        if not filename.endswith(".ucsplit"):
            filename += ".ucsplit"

        ignored_attrs = set(hs.signals.Signal2D([[]]).__dict__.keys())
        ignored_attrs = ignored_attrs.union(ignored_attrs, {"roi", "markers"})

        attrs_to_save = {
            i: j
            for i, j in self.__dict__.items()
            if (i not in ignored_attrs) and not i.startswith("_")
        }
        if hasattr(self, "_preprocessing_type"):
            attrs_to_save["_preprocessing_type"] = self._preprocessing_type
        if hasattr(self, "_tophat_backgrounds"):
            attrs_to_save["_tophat_backgrounds"] = self._preprocessing_type

        with tempfile.TemporaryDirectory() as tmpdir:
            hspy_path = os.path.join(tmpdir, "signal.hspy")
            pkl_path = os.path.join(tmpdir, "state.pkl")
            hs.signals.Signal2D(self).save(hspy_path)
            with open(pkl_path, "wb") as f:
                pickle.dump(attrs_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(hspy_path, arcname="signal.hspy")
                zipf.write(pkl_path, arcname="state.pkl")


def load(filename):

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(filename, "r") as zipf:
            zipf.extractall(tmpdir)

        hspy_path = os.path.join(tmpdir, "signal.hspy")
        pkl_path = os.path.join(tmpdir, "state.pkl")

        raw_signal = hs.load(hspy_path)
        with open(pkl_path, "rb") as f:
            saved_state = pickle.load(f)
        uci = UnitCellImage(
            None,
            uc_centers_matrix=saved_state["uc_centers_matrix"],
            data=raw_signal.data,
        )

        for k, v in saved_state.items():
            setattr(uci, k, v)
        uci.uc_add_markers()

        return uci
