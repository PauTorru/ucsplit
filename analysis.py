import numpy as np
import matplotlib.pyplot as plt
from .utils import add_uci_scale_bar, add_scale_bar

def norm(x):
	return (x-x.min())/(x.max()-x.min())


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

def get_polarization_mod_angle(uci, pxi=None, pyi=None):
	if pxi is None:
		pxi = uci.map_uc_property(polarization_dx)
	if pyi is None:
		pyi = uci.map_uc_property(polarization_dy)

	# Fast 2D magnitude without array stacking
	pmi = np.hypot(pxi, pyi)

	# Angle in degrees [0, 360)
	pai = np.degrees(-np.arctan2(pyi, pxi)) + 180

	return pmi, pai

def add_polar_wheel_key(ax, cmap='twilight'):
	"""Adds a clean polar color wheel inset to an axis."""
	# [x, y, width, height] in axis fractional coordinates
	inset_ax = ax.inset_axes([0.82, 0.82, 0.18, 0.18], projection='polar')

	# Generate 2D ring mesh
	theta = np.linspace(0, 2 * np.pi, 256)
	r = np.linspace(0.6, 1.0, 2)
	Theta, R = np.meshgrid(theta, r)

	# Plot the color wheel ring
	inset_ax.pcolormesh(Theta, R, Theta, cmap=cmap, shading='auto')

	# Styling: hide radial lines and outline
	inset_ax.set_yticklabels([])
	inset_ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
	inset_ax.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=7)
	inset_ax.spines['polar'].set_visible(False)
	return inset_ax

def plot_polarization_mod_angle(uci, pxi=None, pyi=None, **params):
	pmi, pai = get_polarization_mod_angle(uci, pxi, pyi)

	# Side-by-side subplots with modern layout management
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')

	# --- Angle Plot ---
	im1 = ax1.imshow(pai, vmin=0, vmax=360, cmap='twilight')
	ax1.set_title('Polarization Angle')
	add_uci_scale_bar(ax1, uci, **params)

	# Add custom polar wheel overlay inside ax1
	add_polar_wheel_key(ax1, cmap='twilight')

	# --- Modulus Plot ---
	im2 = ax2.imshow(pmi, cmap='viridis')
	ax2.set_title('Polarization Modulus')
	add_uci_scale_bar(ax2, uci, **params)

	cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
	cb2.set_label('px')

	return fig, (ax1, ax2)

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

def add_scale_bar(ax,s,unit_size = 20,unit_name="nm",fontsize = 18,pad=0.1,size_vertical=2, color="white"):
	ax.set_xticks([])
	ax.set_yticks([])
	fontprops = fm.FontProperties(size=fontsize)
	scalebar = AnchoredSizeBar(ax.transData,
						   unit_size/(s.axes_manager[0].scale), str(unit_size)+unit_name, 'lower right', 
						   pad=pad,
						   color=color,
						   frameon=False,
						   size_vertical=size_vertical,
						   fontproperties=fontprops)

	ax.add_artist(scalebar)


def plot_polarization(
	uci,
	dxi=None,
	dyi=None,
	k=1,
	scale=1.0,
	color="yellow",
	head_width=3,
	ax=None,
	**kwargs,
):
	"""Plots a 2D vector field of unit-cell polarization mapped over an image.

	Downsamples both coordinates and polarization vectors via spatial block-averaging.
	"""
	if ax is None:
		ax = plt.gca()

	ax.imshow(uci.original_image, cmap="gray")

	if dxi is None:
		dxi = uci.map_uc_property("polarization_dx")
	if dyi is None:
		dyi = uci.map_uc_property("polarization_dy")

	pos = uci.uc_centers_matrix  # Shape: (Ny, Nx, 2)
	cx = pos[:, :, 0]
	cy = pos[:, :, 1]

	if k > 1:
		# Determine crop dimensions divisible by k to prevent boundary artifacts
		ny, nx = dxi.shape[:2]
		crop_y = (ny // k) * k
		crop_x = (nx // k) * k

		# Block-average coordinates and vector components using block_reduce
		block_size = (k, k)
		red_cx = block_reduce(cx[:crop_y, :crop_x], block_size, np.mean)
		red_cy = block_reduce(cy[:crop_y, :crop_x], block_size, np.mean)
		red_px = block_reduce(dxi[:crop_y, :crop_x], block_size, np.mean)
		red_py = block_reduce(dyi[:crop_y, :crop_x], block_size, np.mean)
	else:
		red_cx, red_cy = cx, cy
		red_px, red_py = dxi, dyi

	# Vector displacement components (negated to match original convention)
	u = -red_px
	v = -red_py

	ax.quiver(
		red_cx,
		red_cy,
		u,
		v,
		color=color,
		scale=1 / scale if scale != 0 else None,
		scale_units="xy",
		angles="xy",
		headwidth=head_width,
		**kwargs,
	)

	ax.set_aspect("equal")
	plt.tight_layout()

from scipy.interpolate import griddata


def plot_over_image(
	im,
	uci,
	t,
	vmin=None,
	vmax=None,
	alpha=0.5,
	cmap="jet",
	interp_method="linear",
	ax=None,
	cbar=True,
):
	"""Overlays a unit-cell property map seamlessly on top of a background image.

	Parameters
	----------
	im : 2D array
		Original grayscale image.
	uci : UnitCellImage object
		Container with unit cell positions (`uc_centers_matrix`).
	t : 2D or 1D array
		Property map values evaluated at unit cell centers.
	vmin, vmax : float, optional
		Colorbar intensity limits.
	alpha : float, optional
		Opacity of the property overlay (0 = image only, 1 = map only).
	cmap : str, optional
		Colormap for the overlaid property map.
	interp_method : {'linear', 'cubic', 'nearest'}, optional
		Spatial interpolation method across unit cell centers.
	ax : matplotlib.axes.Axes, optional
		Target plot axis. Uses current axis if None.
	cbar : bool, optional
		Whether to draw a colorbar.

	Returns
	-------
	cb : Colorbar or None
	"""
	if ax is None:
		ax = plt.gca()

	# Extract coordinates and flatten property values
	x = uci.uc_centers_matrix[..., 0].ravel()
	y = uci.uc_centers_matrix[..., 1].ravel()
	z = np.asarray(t).ravel()

	# Define bounding box / image grid based on target image dimensions
	ny, nx = im.shape[:2]
	grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))

	# Raster interpolation: interpolates scattered UC data into a 2D dense matrix
	t_interp = griddata(
        (x, y), z, (grid_x, grid_y), method=interp_method, fill_value=np.nan
	)

	if vmin is None:
		vmin = np.nanmin(z)
	if vmax is None:
		vmax = np.nanmax(z)

	# 1. Base image (grayscale background)
	ax.imshow(im, cmap="gray", origin="upper")

	# 2. Overlaid property map (single raster layer, extremely lightweight)
	im_overlay = ax.imshow(
		t_interp,
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		alpha=alpha,
		origin="upper",
		interpolation="bilinear",
	)

	ax.set_xticks([])
	ax.set_yticks([])

	cb = None
	if cbar:
		cb = plt.colorbar(im_overlay, ax=ax, fraction=0.046, pad=0.04)

	return cb