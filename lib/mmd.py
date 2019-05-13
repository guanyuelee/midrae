import tensorflow as tf


def get_batch_size(inputs):
    return tf.cast(tf.shape(inputs)[0], tf.float32)

def mmd_penalty(sample_qz, sample_pz):
    shape = sample_qz.get_shape().as_list()
    sample_qz = tf.reshape(sample_qz, shape=[-1, shape[1]*shape[2]*shape[3]])
    sample_pz = tf.reshape(sample_pz, shape=[-1, shape[1]*shape[2]*shape[3]])

    n = get_batch_size(sample_qz)
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = tf.cast((n * n - n) / 2, tf.int32)

    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
    dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2.0 * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
    dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2.0 * dotprods_qz

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2.0 * dotprods


    # Median heuristic for the sigma^2 of Gaussian kernel
    sigma2_k = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
    sigma2_k += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

    res1 = tf.exp( - distances_qz / 2.0 / sigma2_k)
    res1 += tf.exp( - distances_pz / 2.0 / sigma2_k)
    res1 = tf.multiply(res1, 1.0 - tf.eye(n))
    res1 = tf.reduce_sum(res1) / (nf * nf - nf)
    res2 = tf.exp( - distances / 2.0 / sigma2_k)
    res2 = tf.reduce_sum(res2) * 2.0 / (nf * nf)
    stat = res1 - res2  

    return stat


def _mix_rbf_kernel(X, Y, sigmas, wts=None, K_XY_only=False):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0

    XYsqnorm = -2 * XY + c(X_sqnorms) + r(Y_sqnorms)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XY += wt * tf.exp(-gamma * XYsqnorm)

    if K_XY_only:
        return K_XY

    XXsqnorm = -2 * XX + c(X_sqnorms) + r(X_sqnorms)
    YYsqnorm = -2 * YY + c(Y_sqnorms) + r(Y_sqnorms)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XX += wt * tf.exp(-gamma * XXsqnorm)
        K_YY += wt * tf.exp(-gamma * YYsqnorm)

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(tf.shape(K_XX)[0], tf.float32)
    n = tf.cast(tf.shape(K_YY)[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            const_diagonal = tf.cast(const_diagonal, tf.float32)
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2


def mmd2(X, Y, sigma_list=[2.0, 5.0, 10.0, 20.0, 40.0, 80.0], wts=None, const_diagonal=False, biased=False):
    K_XX, K_XY, K_YY, wts_sum = _mix_rbf_kernel(X, Y, sigma_list, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal, biased) # numerics checked at _mmd2 return


def mmd1(X, Y):
    X_mean = tf.reduce_mean(X, axis=0)
    Y_mean = tf.reduce_mean(Y, axis=0)
    return tf.reduce_mean(tf.nn.relu(X_mean - Y_mean))








