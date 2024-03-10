import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def get_subdirs(setup_idx):
    path = './setup-{}/'.format(setup_idx)
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return subdirs

def get_data(setup_idx, subdirs, estimator, bin_size=0, noise_variance=None):
    path = './setup-{}/'.format(setup_idx)

    epoch_data = []
    mi_xt_data = []
    mi_ty_data = []

    if estimator == 'kde':
        if noise_variance is None:
            suffix = 'kde'
        else:
            suffix = 'kde-{}'.format(noise_variance)
    else:
        suffix = bin_size

    for subdir in subdirs:
        fname = path + subdir + '/mi-{}.npz'.format(suffix)
        data = np.load(fname)
        epochs = data['epochs']
        mi_xt = data['mi_xt_epochs']
        mi_yt = data['mi_ty_epochs']

        sorted_idx = np.argsort(epochs)
        mi_xt = mi_xt[sorted_idx]
        mi_yt = mi_yt[sorted_idx]
        epochs = epochs[sorted_idx]

        epoch_data.append(epochs)
        mi_xt_data.append(mi_xt)
        mi_ty_data.append(mi_yt)

    # epoch_data = np.array(epoch_data)
    # mi_xt_data = np.array(mi_xt_data)
    # mi_ty_data = np.array(mi_ty_data)

    return epoch_data, mi_xt_data, mi_ty_data

def get_loss(setup_idx, subdirs):
    train_loss_data = []
    test_loss_data = []
    test_acc_data = []

    for subdir in subdirs:
        path = './setup-{}/'.format(setup_idx)
        fname = path + subdir + '/loss.npz'
        data = np.load(fname)
        train_loss_data.append( data['train_loss'] )
        test_loss_data.append( data['test_loss'] )
        test_acc_data.append( data['test_acc'] )

    return train_loss_data, test_loss_data, test_acc_data

def get_windows(loss_data, threshold=0.65):
    repetitions = len(loss_data)
    window_list = []
    for i in range(repetitions):
        window = [0,10000]
        
        if len(loss_data[i]) == 10000:
            window_list.append(window)
            continue

        start = np.argmin( np.abs( loss_data[i] - threshold ) )

        if start >= 1000:
            window = [start-1000, start+9000]

        if window[1] > len(loss_data[i]):
            window = [len(loss_data[i])-10000, len(loss_data[i])]
            
        window_list.append(window)
        
    return window_list

def average_windowed_data(mi_xt_data, mi_ty_data, window_list):
    
    repetitions = len(window_list)

    windowed_mi_xt_data = []
    windowed_mi_ty_data = []
    for i in range(repetitions):
        window = window_list[i]
        windowed_mi_xt =  mi_xt_data[i][window[0]:window[1]]
        windowed_mi_ty =  mi_ty_data[i][window[0]:window[1]]
        
        windowed_mi_xt_data.append( windowed_mi_xt )
        windowed_mi_ty_data.append( windowed_mi_ty )

    windowed_mi_xt_data = np.array(windowed_mi_xt_data)
    windowed_mi_ty_data = np.array(windowed_mi_ty_data)

    epochs = np.arange(1, windowed_mi_xt_data.shape[1] + 1)
    avg_mi_xt_data = np.mean(windowed_mi_xt_data, axis=0)
    avg_mi_ty_data = np.mean(windowed_mi_ty_data, axis=0)

    return epochs, avg_mi_xt_data, avg_mi_ty_data, windowed_mi_xt_data, windowed_mi_ty_data

def average_data(mi_xt_data, mi_ty_data):
    
    epochs = np.arange(1, mi_xt_data.shape[1] + 1)
    avg_mi_xt_data = np.mean(mi_xt_data, axis=0)
    avg_mi_ty_data = np.mean(mi_ty_data, axis=0)

    return epochs, avg_mi_xt_data, avg_mi_ty_data

def plot_info_plan(mi_xt, mi_yt, epochs, ticks=[], markup=None, max_epoch=None, markup_color='green'):
    """
    Plot the given mutual information values for each layer and each epoch in the information plane.

    Args:
        mi_xt : mutual information between input and layer activations
        mi_yt : mutual information between layer activations and targets
        epochs : epochs at which the mutual information was computed
    """

    mi_xt = np.array( mi_xt )
    mi_yt = np.array( mi_yt )
    epochs = np.array( epochs )

    sorted_idx = np.argsort(epochs)
    mi_xt = mi_xt[sorted_idx]
    mi_yt = mi_yt[sorted_idx]
    epochs = epochs[sorted_idx]

    if max_epoch is not None:
        idx = np.where(epochs <= max_epoch)
        mi_xt = mi_xt[idx]
        mi_yt = mi_yt[idx]
        epochs = epochs[idx]

    n_saved_epochs = mi_xt.shape[0]
    n_layers = mi_xt.shape[1]

    plt.figure()
    
    for l in range(n_layers):
        scatter = plt.scatter(mi_xt[:, l], mi_yt[:, l], c=epochs, cmap='gnuplot', s=30, zorder=3)
    cmap = mpl.cm.get_cmap('gnuplot', n_saved_epochs)
    color = cmap(epochs)

    for e in range(n_saved_epochs):
        plt.plot(mi_xt[e, :], mi_yt[e, :], c=color[e], linewidth=0.5, alpha=0.01)

    cb = plt.colorbar(scatter, ticks=[epochs[0], epochs[-1]])
    cb.ax.set_title('Epochs', fontsize=10)
    cb.set_ticks([epochs[0], epochs[-1]]+ticks)

    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')

    if markup is not None:
        plt.scatter(mi_xt[markup, :], mi_yt[markup, :], c=markup_color, s=50, zorder=3) # marker='x'
        plt.plot(mi_xt[markup, :], mi_yt[markup, :], color=markup_color, linewidth=1)

def plot_path(mi_xt, mi_yt, color='green'):

    plt.scatter(mi_xt, mi_yt, marker='x', c=color, s=60, zorder=3)
    # plt.plot(mi_xt, mi_yt, c=color, linewidth=1)


def pipeline(setup_idx, estimator, subdir_list=np.arange(1,11), bin_size=0, threshold=0.65, noise_variance=None):
    
    path = './setup-{}/'.format(setup_idx)
    
    subdirs = get_subdirs(setup_idx)
    subdirs = [d for d in subdirs if int(d.split('-')[1]) in subdir_list]

    epoch_data, mi_xt_data, mi_ty_data = get_data(setup_idx, subdirs, estimator=estimator, bin_size=bin_size, noise_variance=noise_variance)
    train_loss_data, test_loss_data, test_acc_data = get_loss(setup_idx, subdirs)
    window_list = get_windows(train_loss_data, threshold=threshold)

    epochs, avg_mi_xt_data, avg_mi_ty_data, windowed_mi_xt_data, windowed_mi_ty_data = average_windowed_data(mi_xt_data, mi_ty_data, window_list)

    plot_info_plan(avg_mi_xt_data, avg_mi_ty_data, epochs)
    # plt.savefig(path+"info-plan-{}.png".format('kde' if estimator=='kde' else bin_size), dpi=300, bbox_inches="tight")
    # plt.show()

def plot_loss(train_loss, test_loss, path=None):
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(test_loss, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # plt.savefig(path+"loss.png", dpi=300, bbox_inches="tight")
    # plt.show()

def save(setup_idx, estimator, bin_size=None, noise_variance=None, name="info-plan"):

    if estimator == 'kde':
        if noise_variance is None:
            suffix = 'kde'
        else:
            suffix = 'kde-{}'.format(noise_variance)
    else:
        suffix = bin_size
        
    path = './setup-{}/'.format(setup_idx)

    plt.savefig(path+"{}-{}.png".format(name, suffix), dpi=300, bbox_inches="tight")
    plt.show()