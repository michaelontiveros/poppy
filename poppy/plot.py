import matplotlib.pyplot

def plot(a, title = '', size = 4, cmap = 'twilight_shifted'):
    matplotlib.rc('figure', figsize=(size,size))
    matplotlib.pyplot.matshow(a.reshape((a.shape[0],-1)), cmap = cmap, interpolation = 'none')
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()