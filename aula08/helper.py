import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_images(images, labels = None, n_rows = 1, figsize=(25, 10) ):
    fig = plt.figure(figsize=figsize)
    n_images = len(images)
    for idx in range(n_images):
        ax = fig.add_subplot(n_rows, np.ceil(n_images/n_rows), idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        if (labels is not None):
            ax.set_title(str(labels[idx]))
            ax.title.set_fontsize(16)
            
            
def show_image_in_details(img, fig_size = (12,12)):
    fig = plt.figure(figsize=fig_size) 
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.set_aspect('auto')
    img = img.numpy()
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = np.round(img[x][y],2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',                        
                        color='white' if img[x][y]<thresh else 'black')
            
            
def show_images_rgb(images, labels = None, n_rows = 1, figsize=(25, 10) ):
    fig = plt.figure(figsize=figsize)
    n_images = len(images)
    for idx in range(n_images):
        ax = fig.add_subplot(n_rows, np.ceil(n_images/n_rows), idx+1, xticks=[], yticks=[])
        image_transposed = np.transpose(images[idx], (1,2,0) )
        ax.imshow( image_transposed )
        if (labels is not None):
            ax.set_title(str(labels[idx]))
            ax.title.set_fontsize(16)
            
            
def view_classify(img, ps, **params):
    
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()                        
    
    
import matplotlib.pyplot as plt
import numpy as np
import torch

def make_meshgrid(x, y, h=.001):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = -0.05, 1.05
    y_min, y_max = -0.05, 1.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, model, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    input_data = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float)
    model.eval()
    Z = model(input_data)
    model.train()
    Z = torch.round(Z)
    Z = Z.reshape(xx.shape)    
    out = ax.contourf(xx, yy, Z.detach().numpy(), **params)
    return out, Z
    