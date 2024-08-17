import jax
import matplotlib.pyplot

def plot(a, title = '', size = 4, cmap = 'twilight_shifted'):
    matplotlib.rc('figure', figsize=(size,size))
    a = a.squeeze()
    s = jax.numpy.array(a.shape)
    matplotlib.pyplot.matshow(a.reshape((s[:len(s)//2].prod(),-1)), cmap = cmap, interpolation = 'none')
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()