import tensorflow as tf

def lerp(low, high, alpha):
	"""Linear interpolation"""
	mix = low + (high - low) * alpha
	return mix


def spherical_lerp(low, high, mu):
	"""Spherical interpolation. val has a range of 0 to 1.
	low and high should be unit vertors under l2-norm."""
	shape = mu.get_shape().as_list()
	mu = tf.reshape(mu, shape=[-1, shape[1]*shape[2]*shape[3]])

	shape = low.get_shape().as_list()
	low  = tf.reshape(low,  shape=[-1, shape[1]*shape[2]*shape[3]])
	high = tf.reshape(high, shape=[-1, shape[1]*shape[2]*shape[3]])	

	# ||low|| = ||high|| = 1
	low_norm = tf.nn.l2_normalize(low, axis=[1]) 
	high_norm = tf.nn.l2_normalize(high, axis=[1]) 
	# theta = arccos(low^T * high)
	inner_prod = tf.reduce_sum(tf.multiply(low_norm, high_norm), axis=1, keep_dims=True)
	theta = tf.acos(inner_prod)
	print('theta shape: ', theta.get_shape().as_list())

	mix = tf.sin((1.0-mu)*theta)/tf.sin(theta) * low + tf.sin(mu*theta)/tf.sin(theta) * high
	print('mix shape: ', mix.get_shape().as_list())	

	mix = tf.reshape(mix, shape=[-1, shape[1], shape[2], shape[3]])
	return mix