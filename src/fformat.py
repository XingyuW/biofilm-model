import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.colors import to_rgb
from numpy import linspace
import numpy as np
from PIL import Image
from io import BytesIO
import colorsys
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

_initialized_ = False

def ensure_initialized() -> None:
    global _initialized_
    if not _initialized_:
        __setupFigure__()
        _initialized_ = True


def __setupFigure__():
    font_size = 25
    # parameter
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': font_size,
        'font.weight': 600,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size, 
        "axes.labelweight": 600,
        "axes.titleweight": 600, 
        # set up tick size
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        # set linewidth of spine
        'axes.linewidth': 3,
        "figure.dpi": 300,
        'figure.subplot.left': 0.23,  # single 0.17, twinx 0.14
        'figure.subplot.right': 0.95,  # 0.95, 0.86
        'figure.subplot.bottom': 0.18,  # 0.13, 0.18
        'figure.subplot.top': 0.92,  # 0.9, 0.95
        # Linewidth of plot
        'lines.linewidth': 3,
        'savefig.dpi': 300,
        # 'text.usetex': True
    }
    plt.rcParams.update(params)


# change lightness of color
def color_light(color, lightness):
    # range of color lightness: 0-1, the lower the darker
    rgb_val = to_rgb(color)
    h, _, s = colorsys.rgb_to_hls(*rgb_val)
    return colorsys.hls_to_rgb(h, lightness, s)


cl = color_light


# Darken the color
def darken_color(color, factor=0.5):
    """Darken an RGBA color by scaling the RGB channels."""
    r, g, b, a = color  # Extract RGBA components
    return (r * factor, g * factor, b * factor, a)  # Scale RGB and keep alpha


# Create a darker version of the colormap
def darken_cmap(cmap, factor=0.5):
    """Darken a colormap by multiplying RGB values by a factor."""
    colors = cmap(np.linspace(0, 1, 256))  # Extract RGB values
    darker_colors = colors.copy()
    darker_colors[:, :3] *= factor  # Multiply RGB channels by factor
    return LinearSegmentedColormap.from_list("darker_cmap", darker_colors)


# figure save
def saveFig(fig, name, tif=False):
    if not tif:
        fig.savefig(name+'.png', transparent=True, format='png', dpi=300)  # , bbox_inches="tight")
    else:
        png = BytesIO()
        fig.savefig(png, transparent=True, format='png', dpi=300)  # , bbox_inches="tight")
        with Image.open(png) as png2tif:
            png2tif.save(name+'.tif')


# define color
def color_map(n, cm):
    """
    n: Number of colors
    cm: color map
    """
    if cm=="plasma":
        colors = plt.cm.get_cmap("plasma")(linspace(0, 1, n))
    if cm=="spectral":
        colors = plt.cm.get_cmap("Spectral")(linspace(0, 1, n))
    else:
        colors = plt.cm.get_cmap("viridis")(linspace(0, 1, n))
        
    return colors


# define random color map
def array_color_map(x,y, cmap="plasma", darken=1):
    """
    x: array of x values
    y: array of values
    cmap: color map
    """
    color_map = matplotlib.colormaps[cmap]
    color_map = darken_cmap(color_map, darken)
    # Normalize based on y values
    norm = Normalize(vmin=np.min(y), vmax=np.max(y))
    # Create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Convert ndarray to python list
    segments_list = [seg for seg in segments]
    # Create a LineCollection
    lc = LineCollection(segments_list, cmap=color_map, norm=norm)
    lc.set_array(y)  # Map y values to the colormap
    return lc


# convert RGB within the range of 0 and 1
def rgb_scale(rgb_list):
    return rgb_list[0]/255, rgb_list[1]/255, rgb_list[2]/255


# Scientific notation for LaTeX
def sci_notation_tex(number, unit, precision=2):
    if not np.isfinite(number):
        return r"$\infty$" if number > 0 else r"$-\infty$" if number < 0 else r"$\mathrm{NaN}$"
    if number == 0:
        return r"$0$"

    exponent = int(np.floor(np.log10(abs(number))))
    coeff = round(number / 10**exponent, precision)
    sign = "-" if number < 0 else ""

    if exponent == 0:
        return r"${}{}$ {}".format(sign, coeff, unit)
    elif exponent == 1:
        return r"${}{}$ {}".format(sign, coeff*10, unit)
    else:
        return r"${}{} \times 10^{{{}}}$ {}".format(sign, coeff, exponent, unit)


# marker
mkr = MarkerStyle("o", "none")
tmkr = MarkerStyle("^", "none")
smkr = MarkerStyle("s", "none")

# Scatter parameters
ms = 5  # 100
mlw = 0.5  # 2.5

# figsize for the paper
figsize = (10, 6.25)
