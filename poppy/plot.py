import jax
import matplotlib.pyplot
from poppy.constant import CMAP

def plot(a, title = '', size = 4, dpi = 256, cmap = CMAP):
    matplotlib.rc('figure', figsize = (size,size), dpi = dpi)
    a = a.squeeze()
    s = jax.numpy.array(a.shape)
    matplotlib.pyplot.matshow(a.reshape((s[:len(s)//2].prod(),-1)), cmap = cmap, interpolation = 'none')
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()
