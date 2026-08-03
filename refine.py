import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar
from skimage.morphology import disk
from scipy.ndimage import white_tophat

@njit(fastmath=True)
def nb_refine_single_atom_com(px, py, image, radius, iters):
	"""
	Refines a single atom coordinate using local Center of Mass.
	No mask allocation, natively compiled.
	"""
	ny, nx = image.shape
	current_x = px
	current_y = py

	for _ in range(iters):
		# Establish the bounding box
		x_start = int(np.floor(current_x - radius))
		x_end = int(np.ceil(current_x + radius)) + 1
		y_start = int(np.floor(current_y - radius))
		y_end = int(np.ceil(current_y + radius)) + 1

		# Clip boundaries to keep them safely inside the image matrix
		if x_start < 0: x_start = 0
		if x_end > nx: x_end = nx
		if y_start < 0: y_start = 0
		if y_end > ny: y_end = ny

		sum_intensity = 0.0
		sum_x = 0.0
		sum_y = 0.0
		r2 = radius * radius

		# Iterate through the local window, evaluating the disk mathematically
		for y_idx in range(y_start, y_end):
			dy = y_idx - current_y
			for x_idx in range(x_start, x_end):
				dx = x_idx - current_x
				
				# Check if the pixel falls inside the circle radius
				if (dx * dx + dy * dy) <= r2:
					intensity = image[y_idx, x_idx]
					sum_intensity += intensity
					sum_x += intensity * x_idx
					sum_y += intensity * y_idx

		# Update coordinates if we have a valid denominator
		if sum_intensity > 0.0:
			current_x = sum_x / sum_intensity
			current_y = sum_y / sum_intensity
		else:
			break

	return current_x, current_y

@njit(parallel=True, fastmath=True)
def nb_refine_all_cells_com(pos_data, ucs_data, radius, iters):
	"""
	Parallel loop executing COM corrections across every row, column, 
	and atom index simultaneously across all CPU threads.
	"""
	n_rows, n_cols, n_atoms, _ = pos_data.shape
	refined_pos = pos_data.copy()

	# prange distributes these iterations across your CPU cores
	for r in prange(n_rows):
		for c in prange(n_cols):
			cell_image = ucs_data[r, c, :, :]
			img_min = np.min(cell_image)
			img_max = np.max(cell_image)
			norm_image = (cell_image - img_min) / (img_max - img_min)
			
			for a in range(n_atoms):
				px = pos_data[r, c, a, 0]
				py = pos_data[r, c, a, 1]
				
				rx, ry = nb_refine_single_atom_com(px, py, norm_image, radius, iters)
				
				refined_pos[r, c, a, 0] = rx
				refined_pos[r, c, a, 1] = ry

	return refined_pos


@njit(fastmath=True)
def nb_eval_gaussians_2d(nx, ny, params, n_peaks):
	"""Evaluates N 2D Gaussians on a grid of size (ny, nx)."""
	Z = np.zeros((ny, nx))
	for i in range(n_peaks):
		x0, y0, A, sx, sy, theta = params[i*6 : i*6+6]
		cos_t = np.cos(theta)
		sin_t = np.sin(theta)
		
		for y in range(ny):
			dy = y - y0
			for x in range(nx):
				dx = x - x0
				x_rot = dx * cos_t + dy * sin_t
				y_rot = -dx * sin_t + dy * cos_t
				
				exp_val = -0.5 * ((x_rot / sx)**2 + (y_rot / sy)**2)
				if exp_val < -700.0: exp_val = -700.0
				Z[y, x] += A * np.exp(exp_val)
	return Z


@njit(fastmath=True)
def nb_fit_single_cell_2dgauss(cell_image, init_params, n_peaks, max_drift, max_iter=200, tol=1e-4):
	"""Levenberg-Marquardt solver for N overlapping Gaussians."""
	ny, nx = cell_image.shape
	M = ny * nx
	num_params = 6 * n_peaks
	p = init_params.copy()
	p0_static = init_params.copy()
	max_width = nx / 2.0
	lam = 0.01
	
	J_base = np.empty((num_params, M))
	J = J_base.T
	cell_flat = cell_image.ravel()
	sqrt_weights = np.sqrt(np.maximum(cell_flat + 1e-3, 1e-8))
	
	for _ in range(max_iter):
		# 1. Evaluate current model
		f = nb_eval_gaussians_2d(nx, ny, p, n_peaks)
		residual = cell_image.ravel() - f.ravel()
		weighted_residual = sqrt_weights * residual

		# 2. Build Analytical Jacobian

		for i in range(n_peaks):
			idx = i * 6
			x0, y0, A, sx, sy, theta = p[idx:idx+6]
			
			sx = max(sx, 1e-4)
			sy = max(sy, 1e-4)
			
			cos_t = np.cos(theta)
			sin_t = np.sin(theta)
			
			inv_sx2 = 1.0 / (sx * sx)
			inv_sy2 = 1.0 / (sy * sy)
			inv_sx3 = inv_sx2 / sx
			inv_sy3 = inv_sy2 / sy
			idx_m = 0
			for y in range(ny):
				dy = y - y0
				dy_sin = dy * sin_t
				dy_cos = dy * cos_t
				for x in range(nx):
					dx = x - x0
					
					x_rot = dx * cos_t + dy_sin
					y_rot = -dx * sin_t + dy_cos
					exp_val = -0.5 * ((x_rot * x_rot) * inv_sx2 + (y_rot * y_rot) * inv_sy2)
					if exp_val < -700.0: 
						exp_val = -700.0

					exp_res = np.exp(exp_val)
					Z_comp = A * exp_res
					
					d_ex = x_rot * inv_sx2
					d_ey = y_rot * inv_sy2

					dz_dx0 = Z_comp * (d_ex * cos_t - d_ey * sin_t)
					dz_dy0 = Z_comp * (d_ex * sin_t + d_ey * cos_t)
					dz_dA  = exp_res
					dz_dsx = Z_comp * (x_rot * x_rot) * inv_sx3
					dz_dsy = Z_comp * (y_rot * y_rot) * inv_sy3
					dz_dt  = Z_comp * (d_ex * (-y_rot) + d_ey * x_rot)
					sw = sqrt_weights[idx_m]
					
					J[idx_m, idx]   = sw * dz_dx0
					J[idx_m, idx+1] = sw * dz_dy0
					J[idx_m, idx+2] = sw * dz_dA
					J[idx_m, idx+3] = sw * dz_dsx
					J[idx_m, idx+4] = sw * dz_dsy
					J[idx_m, idx+5] = sw * dz_dt
					
					idx_m += 1
				
		# 3. LM Step Calculation
		JT = J.T
		H = np.dot(JT, J)
		gradient = np.dot(JT, residual)
		
		for i in range(num_params):
			H[i, i] += lam * H[i, i] + 1e-10
			
		try:
			step = np.linalg.solve(H, gradient)
		except:
			break
			
		p += step

		# 4. Enforce physical boundaries
		for i in range(n_peaks):
			idx = i * 6
			
			p[idx]   = max(p0_static[idx] - max_drift, min(p0_static[idx] + max_drift, p[idx]))
			p[idx+1] = max(p0_static[idx+1] - max_drift, min(p0_static[idx+1] + max_drift, p[idx+1]))
			
			p[idx+2] = max(0.0, p[idx+2])
			p[idx+3] = max(0.1, min(max_width, p[idx+3]))
			p[idx+4] = max(0.1, min(max_width, p[idx+4]))

		if np.max(np.abs(step)) < tol:
			break

	return p

@njit(parallel=True, fastmath=True, nogil=True)
def nb_fit_all_cells_2dgauss(pos_data, preprocessed_ucs, f, iters,tol, progress_hook, max_drift):
	"""Parallel wrapper to initialize and fit every unit cell."""
	n_rows, n_cols, n_atoms, _ = pos_data.shape
	ny, nx = preprocessed_ucs.shape[2], preprocessed_ucs.shape[3]
	num_params = 6 * n_atoms
	
	# Store results
	fitted_params = np.zeros((n_rows, n_cols, n_atoms, 6))
	
	# Base initialization logic for widths based on fraction 'f'
	avg_dim = (nx + ny) / 2.0
	sx_init = (f + 0.01) * avg_dim
	sy_init = (f - 0.01) * avg_dim

	scale = 100.0

	for r in prange(n_rows):
		for c in range(n_cols):
			img_max = np.max(preprocessed_ucs[r, c, :, :])
			scale_factor = target_scale / (img_max + 1e-8)
			cell_image = scale_factor*preprocessed_ucs[r, c, :, :]
			p0 = np.zeros(num_params)
			
			# Build initial guesses array
			for a in range(n_atoms):
				px = pos_data[r, c, a, 0]
				py = pos_data[r, c, a, 1]
				
				# Estimate initial amplitude
				xi, yi = int(np.round(px)), int(np.round(py))
				if xi < 0: xi = 0
				if xi >= nx: xi = nx - 1
				if yi < 0: yi = 0
				if yi >= ny: yi = ny - 1
				A_init = cell_image[yi, xi]
				
				idx = a * 6
				p0[idx]   = px
				p0[idx+1] = py
				p0[idx+2] = A_init
				p0[idx+3] = sx_init
				p0[idx+4] = sy_init
				p0[idx+5] = 0.0  # Rotation init
				
			# Run JIT Optimization
			p_fit = nb_fit_single_cell_2dgauss(cell_image, p0, n_atoms, max_drift, max_iter=iters,tol=tol)
			
			# Map flat parameter array back to structured output
			for a in range(n_atoms):
				idx = a * 6
				for param_idx in range(6):
					fitted_params[r, c, a, param_idx] = p_fit[idx + param_idx]
				fitted_params[r, c, a, 2] = p_fit[idx + 2]/scale_factor
			progress_hook.update(1)
					
	return fitted_params


class RefinePositions:

	def refine_atom_poisitions_com(self,iters=5,radius=5):
		self.check_pos_data()
		pos_matrix = self.pos_data.astype(np.float64) 
		ucs_matrix = self.data
		print("Executing multi-core Numba Center-of-Mass position refinement...")
		refined_positions = nb_refine_all_cells_com(
			pos_matrix, 
			ucs_matrix, 
			float(radius), 
			int(iters)
		)
		self.pos_data = refined_positions
		print("COM refinement completed successfully.")
		self.uc_add_markers()

	def default_preprocessing(self, ucs_matrix):
		"""
		Subtracts the minimum of each unit cell to create a zero-baseline image.
		Returns the processed data and the tracked baseline state.
		"""
		# Calculate minimums over the spatial axes (ny, nx)
		baselines = np.min(ucs_matrix, axis=(2, 3), keepdims=True)
		processed_data = ucs_matrix - baselines
		
		# Squeeze baselines for easier storage: (n_rows, n_cols)
		self._default_baselines = baselines.squeeze()
		return processed_data

	def inverse_default_preprocessing(self, modeled_ucs):
		"""
		Re-adds the stored baseline states to the mathematically evaluated pure Gaussians.
		"""
		# Expand baselines back to (n_rows, n_cols, 1, 1) for broadcasting
		return modeled_ucs + self._default_baselines[:, :, np.newaxis, np.newaxis]

	def preprocess_tophat(self, ucs_matrix, radius=5, **kwargs):
		"""
		Applies a 2D Top-Hat filter to every unit cell.
		Isolates peaks by subtracting a structural rolling-ball background.
		"""
		
		# Track the extracted background so the inverse function can restore it
		# White Top-Hat = Original - Opened Image -> Background = Original - White Top-Hat
		processed_data = np.zeros_like(ucs_matrix)
		n_rows, n_cols = ucs_matrix.shape[0], ucs_matrix.shape[1]
		
		# Define a circular footprint based on the provided radius parameter
		y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
		footprint = x**2 + y**2 <= radius**2
		
		for r in range(n_rows):
			for c in range(n_cols):
				processed_data[r, c] = white_tophat(ucs_matrix[r, c], footprint=footprint)
				
		# Structural background array is what was eliminated by the filter
		self._tophat_backgrounds = ucs_matrix - processed_data
		return processed_data

	def inverse_preprocess_tophat(self, modeled_ucs):
		"""
		Re-adds the complex 2D structural background back to the modeled Gaussians.
		"""
		return modeled_ucs + self._tophat_backgrounds

	def refine_uc_atoms_2dgauss(self, col_width_ratio=0.2, iters=30,tol=1e-4,preprocessing="default", max_drift = 10, **kwargs):
		self.check_pos_data()
		
		pos_matrix = self.pos_data.astype(np.float64)
		ucs_matrix = self.data
		preprocessing_fs = {
			"default": self.default_preprocessing,
			"tophat": self.preprocess_tophat
		}
		if preprocessing not in preprocessing_fs:
			raise ValueError(f"Unknown preprocessing type '{preprocessing}'")

		self._preprocessing_type = preprocessing
		
		preprocessed_data = preprocessing_fs[self._preprocessing_type](
			ucs_matrix, **kwargs
		)

		total_cells = pos_matrix.shape[0] * pos_matrix.shape[1]
		with ProgressBar(total=total_cells) as numba_progress_bar:
			self.gaussian_params = nb_fit_all_cells_2dgauss(
				pos_matrix, 
				preprocessed_data, 
				float(col_width_ratio), 
				int(iters),
				tol,
				numba_progress_bar,
				max_drift
			)
		self.pos_data[:, :, :, 0] = self.gaussian_params[:, :, :, 0]
		self.pos_data[:, :, :, 1] = self.gaussian_params[:, :, :, 1]
		self.uc_add_markers()

	def eval_uc_model(self):
		"""
		Evaluates the fitted parameters back into image arrays, 
		and reconstructs them through the inverse preprocessing pipeline.
		"""
		if not hasattr(self, 'gaussian_params'):
			raise ValueError("Must run refine_uc_atoms_2dgauss() before evaluating.")
			
		n_rows, n_cols, n_atoms, _ = self.gaussian_params.shape
		ny, nx = self.data.shape[2], self.data.shape[3]
		
		modeled_data = np.zeros((n_rows, n_cols, ny, nx))
		
		# Generate pure models (we don't need JIT here as eval is relatively fast, 
		# but we reuse the JIT math function for efficiency)
		for r in range(n_rows):
			for c in range(n_cols):
				# Flatten params from (n_atoms, 6) to a 1D array of length n_atoms*6
				flat_params = self.gaussian_params[r, c].ravel()
				modeled_data[r, c] = nb_eval_gaussians_2d(nx, ny, flat_params, n_atoms)
				
		inverse_fs = {
			"default": self.inverse_default_preprocessing,
			"tophat": self.inverse_preprocess_tophat
		}
		reconstructed_data = inverse_fs[self._preprocessing_type](modeled_data)

		return reconstructed_data