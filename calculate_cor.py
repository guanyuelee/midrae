# this file is used for plotting interpolation of a specified dataset

# Given four 2-D latent points, if they form a convex quadrilateral, then we use mesh to
# approximate the manifold of the latent space

import numpy as np
from absl import flags
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from absl import app
from lib import data, utils
import all_aes
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle as pkl

from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show

FLAGS = flags.FLAGS
flags.DEFINE_integer('n_int', 10, 'number of interpolations between two points.')
flags.DEFINE_string('img_dir', '.', 'Directory to save our images.')
flags.DEFINE_string('ae_dir', '.', 'Directory of autoencoder model.')
flags.DEFINE_string('dataset', 'mnist32', 'Name of dataset')
flags.DEFINE_bool('acai', True, 'Whether to test covariance of distribution in ACAI.')
flags.DEFINE_bool('miae', True, 'Whether to test covariance of distribution in MI-AE.')
flags.DEFINE_bool('midrae', False, 'Whether to test covariance of distribution in midrae.')
flags.DEFINE_integer('dim', 32, 'Dimension of latent representation.')
FLAGS = flags.FLAGS

def get_latents_and_labels(sess, ops, dataset, batches=None):
    batch = FLAGS.batch
    with tf.Graph().as_default():
        data_in = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess_new:
            images = []
            labels = []
            while True:
                try:
                    payload = sess_new.run(data_in)
                    images.append(payload['x'])
                    assert images[-1].shape[0] == 1 or batches is not None
                    labels.append(payload['label'])
                    if len(images) == batches:
                        break
                except tf.errors.OutOfRangeError:
                    break
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    latents = [sess.run(ops.encode,
                        feed_dict={ops.x: images[p:p + batch]})
               for p in range(0, images.shape[0], FLAGS.batch)]
    latents = np.concatenate(latents, axis=0)
    shape = latents.shape
    latents = latents.reshape([latents.shape[0], -1])
    return images, latents, labels, shape


def get_images_and_latents(dir, dataset, batch):
    ae, ds = utils.load_ae(dir, dataset, batch, all_aes.ALL_AES, return_dataset=True)
    with utils.HookReport.disable():
        ae.eval_mode()
    if dataset == 'celeba32':
        images, latents, labels, shape = get_latents_and_labels(ae.eval_sess, ae.eval_ops, ds.test, batches=batch)
    elif dataset == 'lines2':
        images, latents, labels, shape = get_latents_and_labels(ae.eval_sess, ae.eval_ops, ds.test, batches=batch)
    else:
        images, latents, labels, shape = get_latents_and_labels(ae.eval_sess, ae.eval_ops, ds.train_once, batches=None)

    return images, latents, labels, shape


def get_specified_idx(labels, spec, n_lines, nth=10):
    n = len(spec)
    if n < n_lines:
        for i in range(n_lines - n):
            spec.append([np.random.choice(nth, 1)[0], np.random.choice(nth, 1)[0]])
    n = len(spec)
    idx = []
    sorted_idx = np.linspace(0, labels.shape[0] - 1, labels.shape[0]).astype(np.int32)
    for i in range(n):
        cand1 = sorted_idx[labels == spec[i][0]]
        cand2 = sorted_idx[labels == spec[i][1]]
        s_idx1 = cand1[np.random.choice(cand1.shape[0], 1)[0]]
        s_idx2 = cand2[np.random.choice(cand2.shape[0], 1)[0]]
        idx.append([s_idx1, s_idx2])

    return idx


def get_encode(sess, ops, images, batch):
    print(images.shape)
    latents = [sess.run(ops.encode,
                        feed_dict={ops.x: images[p:min(p + batch, images.shape[0])]})
               for p in range(0, images.shape[0], batch)]
    latents = np.concatenate(latents, axis=0)
    latents = latents.reshape([latents.shape[0], -1])
    return latents


def get_decode(sess, ops, latents, batch, shape):
    print(latents.shape)
    latents = np.reshape(latents, [-1, shape[1], shape[2], shape[3]])
    latents = [sess.run(ops.decode,
                        feed_dict={ops.h: latents[p:min(p + batch, latents.shape[0])]})
               for p in range(0, latents.shape[0], batch)]
    latents = np.concatenate(latents, axis=0)
    return latents


def get_interpolated(type, n_int, latent, idx, sorted=False, rand_seeds=None):
    idx_np = np.array(idx, dtype=np.int32)
    n = idx_np.shape[0]
    latents = np.zeros(shape=[n, n_int + 2, latent.shape[1]], dtype=np.float32)

    if type == 'linear':
        if not sorted:
            sorted_list = np.random.rand(n_int+2)
        else:
            sorted_list = np.linspace(1.0, 0.0, n_int+2)

        if rand_seeds is not None:
            sorted_list = rand_seeds

        for i in range(n):
            x1 = latent[idx_np[i, 0]]
            x2 = latent[idx_np[i, 1]]
            latents[i, 0] = x1
            latents[i, -1] = x2
            for j in range(n_int):
                latents[i, j+1] = x1 * sorted_list[j+1] + x2 * (1 - sorted_list[j+1])
        rand_seeds = sorted_list
        return latents, rand_seeds
    else:
        if not sorted:
            sorted_list = np.random.rand(n_int+2, latent.shape[1])
        else:
            sorted_list = np.linspace(1.0, 0.0, n_int+2)

        if rand_seeds is not None:
            sorted_list = rand_seeds

        for i in range(n):
            x1 = latent[idx_np[i, 0]]
            x2 = latent[idx_np[i, 1]]
            latents[i, 0] = x1
            latents[i, -1] = x2
            for j in range(n_int):
                if not sorted:
                    r = np.random.rand(x1.shape[0])
                    latents[i, j + 1] = np.multiply(x1, r) + np.multiply(x2, (1 - r))
                else:
                    latents[i, j + 1] = x1 * sorted_list[j + 1] + x2 * (1 - sorted_list[j + 1])
        return latents, sorted_list

def normalize(x):
    x = (x - x.min())/(x.max() - x.min())
    return x

def cal_corrcoef(latent):
    print(latent.shape)
    corrcoef = np.corrcoef(latent)
    print(corrcoef.shape)
    return corrcoef


def get_percentage_close(latents, percent):
    means = np.mean(latents, axis=0)
    distance = np.sqrt(np.sum(np.square(latents - means), axis=1))
    sort_idx = np.argsort(distance, axis=0)
    n = sort_idx.shape[0]
    m = np.int(percent * n)
    latent_good = latents[sort_idx[:m]]
    return latent_good

def normalize_2d(latent):
    latent = latent - np.mean(latent, axis=0)
    xx = np.max(latent, axis=0)
    mm = np.min(latent, axis=0)
    norm = np.divide((latent - mm),(xx - mm))
    return norm

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs

def get_vector(scale, eigen):
    A = eigen.copy()
    A[:, 0] = eigen[:, 0] * scale[0]
    A[:, 1] = eigen[:, 1] * scale[1]
    return A


# make my application which can read random four images or images whose parameters is choosen by you.
def main(argv):
    del argv
    width = 32
    height = 32

    acai = FLAGS.acai
    miae = FLAGS.miae
    midrae = FLAGS.midrae
    dataset = FLAGS.dataset
    dim = FLAGS.dim

    if dataset == 'mnist32' and dim == 32:
        ae_dir_acai = './TRAIN/mnist32/ACAI_advdepth16_advweight0.5_depth16_latent2_reg0.2_scales3' if acai else None
        ae_dir_miae = './TRAIN/mnist32/MIAE_advdepth16_advweight0.5_depth16_latent2_reg0.2_scales3' if miae else None
        ae_dir_midrae = './TRAIN/mnist32/MIDRAE_advdepth16_advfake0.05_advnoise1.0_advweight0.5_depth16_latent2_reg0.2_scales3_wgt_mmd1.0' if midrae else None

    elif dataset == 'mnist32' and dim == 256:
        ae_dir_acai = './TRAIN/mnist32/ACAI_advdepth16_advweight0.5_depth16_latent16_reg0.2_scales3' if acai else None
        ae_dir_miae = './TRAIN/mnist32/MIAE_advdepth16_advweight0.5_depth16_latent16_reg0.2_scales3' if miae else None
        ae_dir_midrae = './TRAIN/mnist32/MIDRAE_advdepth16_advfake0.05_advnoise1.0_advweight0.5_depth16_latent2_reg0.2_scales3_wgt_mmd1.0' if midrae else None

    elif dataset == 'svhn32' and dim == 32:
        ae_dir_acai = './TRAIN/svhn32/ACAI_advdepth64_advweight0.5_depth64_latent2_reg0.2_scales3' if acai else None
        ae_dir_miae = './TRAIN/svhn32/MIAE_advdepth64_advweight0.5_depth64_latent2_reg0.2_scales3' if miae else None
        ae_dir_midrae = './TRAIN/svhn32/MIDRAE_advdepth64_advfake0.05_advnoise0.5_advweight0.5_depth64_latent2_reg0.2_scales3_wgt_mmd1.0' if midrae else None

    elif dataset == 'svhn32' and dim == 256:
        ae_dir_acai = './TRAIN/svhn32/ACAI_advdepth64_advweight0.5_depth64_latent16_reg0.2_scales3' if acai else None
        ae_dir_miae = './TRAIN/svhn32/MIAE_advdepth64_advweight0.5_depth64_latent16_reg0.2_scales3'if miae else None
        ae_dir_midrae = './TRAIN/svhn32/MIDRAE_advdepth64_advfake0.05_advnoise0.5_advweight0.5_depth64_latent2_reg0.2_scales3_wgt_mmd1.0'if midrae else None

    elif dataset == 'cifar10' and dim == 256:
        ae_dir_acai = './TRAIN/cifar10/ACAI_advdepth64_advweight0.5_depth64_latent16_reg0.2_scales3'if acai else None
        ae_dir_miae = './TRAIN/cifar10/MIAE_advdepth64_advweight0.5_depth64_latent16_reg0.2_scales3'if miae else None
        ae_dir_midrae = './TRAIN/cifar10/MIDRAE_advdepth64_advfake0.03_advnoise0.01_advweight0.5_depth64_latent16_reg0.2_scales3_wgt_mmd1.0'if midrae else None

    elif dataset == 'cifar10' and dim == 1024:
        ae_dir_acai = './TRAIN/cifar10/ACAI_advdepth64_advweight0.5_depth64_latent64_reg0.2_scales3'if acai else None
        ae_dir_miae = './TRAIN/cifar10/MIAE_advdepth64_advweight0.5_depth64_latent64_reg0.2_scales3'if miae else None
        ae_dir_midrae = './TRAIN/cifar10/MIDRAE_advdepth64_advfake0.03_advnoise0.01_advweight0.5_depth64_latent16_reg0.2_scales3_wgt_mmd1.0'if midrae else None

    elif dataset == 'celeba32' and dim == 32:
        ae_dir_acai = './TRAIN/celeba32/ACAI_advdepth64_advweight0.5_depth64_latent2_reg0.2_scales3'if acai else None
        ae_dir_miae = './TRAIN/celeba32/MIAE_advdepth64_advweight0.5_depth64_latent2_reg0.2_scales3_wgt_mmd1.0'if miae else None
        ae_dir_midrae = './TRAIN/celeba32/MIDRAE_advdepth64_advfake0.01_advnoise0.1_advweight0.5_depth64_latent2_reg0.2_scales3_wgt_mmd1.0'if midrae else None

    elif dataset == 'celeba32' and dim == 256:
        ae_dir_acai = './TRAIN/celeba32/ACAI_advdepth64_advweight0.5_depth64_latent16_reg0.2_scales3'if acai else None
        ae_dir_miae = './TRAIN/celeba32/MIAE_advdepth64_advweight0.5_depth64_latent16_reg0.2_scales3'if miae else None
        ae_dir_midrae = './TRAIN/celeba32/MIDRAE_advdepth64_advfake0.01_advnoise0.1_advweight0.5_depth64_latent2_reg0.2_scales3_wgt_mmd1.0'if midrae else None

    elif dataset == 'lines2':
        advweight = 2.0
        ae_dir_acai = './TRAIN/lines2/ACAI_advdepth16_advweight%f_depth16_latent2_reg0.2_scales5' % advweight if acai else None
        ae_dir_miae = './TRAIN/lines2/MIAE_advdepth16_advweight%f_depth16_latent2_reg0.2_scales5' % advweight if miae else None
        ae_dir_midrae = './TRAIN/lines2/MIDRAE_advdepth64_advfake0.01_advnoise0.1_advweight0.5_depth64_latent2_reg0.2_scales3_wgt_mmd1.0' if midrae else None

    else:
        raise NotImplementedError

    batch = 4000

    n_int = 9
    n_lines = 10

    if dataset == 'mnist32':
        channel = 1
    else:
        channel = 3

    image = np.zeros([n_lines*3*height, (n_int+2)*width, channel], dtype=np.float32)

    images, latent_acai, labels, shape = get_images_and_latents(ae_dir_acai, dataset, batch)
    corr_acai = cal_corrcoef(latent_acai.T)

    if miae:
        images, latent_macai, labels, shape = get_images_and_latents(ae_dir_miae, dataset, batch)
        corr_macai = cal_corrcoef(latent_macai.T)

    if midrae:
        images, latent_wacai, labels, shape = get_images_and_latents(ae_dir_midrae, dataset, batch)

    print(corr_acai)
    print(corr_macai)


if __name__ == '__main__':
    app.run(main)
