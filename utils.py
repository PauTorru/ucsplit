import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


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