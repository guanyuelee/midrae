# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/usr/bin/env python
"""
Add Multidimensional interpolation and Dual Regularization to ACAI.
"""
# Add fake

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

import tensorflow as tf
from lib import data, layers, train, utils, classifiers, eval
from lib.mmd import mmd2
import numpy as np

FLAGS = flags.FLAGS


class MIDRAE(train.AE):

    def model(self, latent, depth, scales, advweight, advdepth, reg, advnoise, advfake, wgt_mmd):
        ## define inputs
        x = tf.placeholder(tf.float32,
                           [None, self.height, self.width, self.colors], 'x')
        l = tf.placeholder(tf.float32, [None, self.nclass], 'label')
        h = tf.placeholder(tf.float32, [None, self.height >> scales, self.width >> scales, latent], 'h')

        def encoder(x):
            return layers.encoder(x, scales, depth, latent, 'ae_enc')

        def decoder(h):
            v = layers.decoder(h, scales, depth, self.colors, 'ae_dec')
            return v

        def disc(x):
            # return tf.reduce_mean(layers.encoder(x, scales, advdepth, latent, 'disc'), axis=[1, 2, 3])
            y = layers.encoder(x, scales, depth, latent, 'disc')
            return y

        encode = encoder(x)
        ae = decoder(encode)
        loss_ae = tf.losses.mean_squared_error(x, ae)

        decode = decoder(h)

        ## impose regularization on latent space
        encode_flat = tf.reshape(encode, [tf.shape(encode)[0], -1])
        h_flat = tf.reshape(h, [tf.shape(h)[0], -1])
        loss_mmd = tf.nn.relu(mmd2(encode_flat, h_flat))

        ## impose regularization on latent space
        alpha_mix = tf.random_uniform(tf.shape(encode), 0, 1)
        alpha_mix = 0.5 - tf.abs(alpha_mix - 0.5)  # Make interval [0, 0.5]
        encode_mix = alpha_mix * encode + (1 - alpha_mix) * encode[::-1]
        decode_mix = decoder(encode_mix)

        loss_disc_real = tf.reduce_mean(tf.square(disc(ae + reg * (x - ae))))
        loss_disc_mix = tf.reduce_mean(tf.square(disc(decode_mix) - alpha_mix))
        loss_ae_disc_mix = tf.reduce_mean(tf.square(disc(decode_mix)))

        alpha_noise = tf.random_uniform(tf.shape(encode), 0, 1)
        encode_mix_noise = alpha_noise * encode + (1 - alpha_noise) * h
        decode_mix_noise = decoder(encode_mix_noise)

        loss_disc_noise = tf.reduce_mean(tf.square(disc(decode_mix_noise) - alpha_noise))
        loss_ae_disc_noise = tf.reduce_mean(tf.square(disc(decode_mix_noise)))

        alpha_fake = 0.5  # I think here we can have another try.
        loss_disc_fake = tf.reduce_mean(tf.square(disc(decode) - alpha_fake))
        loss_ae_disc_fake = tf.reduce_mean(tf.square(disc(decode)))

        utils.HookReport.log_tensor(loss_ae, 'loss_ae')
        utils.HookReport.log_tensor(loss_disc_real, 'loss_disc_real')
        utils.HookReport.log_tensor(loss_disc_mix, 'loss_disc_mix')
        utils.HookReport.log_tensor(loss_ae_disc_mix, 'loss_ae_disc_mix')
        utils.HookReport.log_tensor(loss_disc_noise, 'loss_disc_noise')
        utils.HookReport.log_tensor(loss_ae_disc_noise, 'loss_ae_disc_noise')
        utils.HookReport.log_tensor(loss_disc_fake, 'loss_disc_fake')
        utils.HookReport.log_tensor(loss_ae_disc_fake, 'loss_ae_disc_fake')
        utils.HookReport.log_tensor(loss_mmd, 'loss_mmd')

        xops = classifiers.single_layer_classifier(tf.stop_gradient(encode), l, self.nclass)
        xloss = tf.reduce_mean(xops.loss)
        utils.HookReport.log_tensor(xloss, 'classify_latent')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ae_vars = tf.global_variables('ae_')
        disc_vars = tf.global_variables('disc')
        xl_vars = tf.global_variables('single_layer_classifier')

        with tf.control_dependencies(update_ops):
            train_ae = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_ae + advweight * loss_ae_disc_mix + advnoise * loss_ae_disc_noise
                + advfake * loss_ae_disc_fake + wgt_mmd * loss_mmd,
                var_list=ae_vars)
            train_d = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                loss_disc_real + loss_disc_mix + loss_disc_noise + loss_disc_fake,
                var_list=disc_vars)
            train_xl = tf.train.AdamOptimizer(FLAGS.lr).minimize(
                xloss,
                global_step=tf.train.get_global_step(),
                var_list=xl_vars)

        ops = train.AEOps(x, h, l, encode, decode, ae,
                          tf.group(train_ae, train_d, train_xl),
                          train_xl,
                          classify_latent=xops.output)

        n_interpolations = 16
        n_images_per_interpolation = 16

        def gen_images():
            return self.make_sample_grid_and_save(
                ops, interpolation=n_interpolations,
                height=n_images_per_interpolation)

        recon, inter, slerp, samples = tf.py_func(
            gen_images, [], [tf.float32] * 4)
        tf.summary.image('reconstruction', tf.expand_dims(recon, 0))
        tf.summary.image('interpolation', tf.expand_dims(inter, 0))
        tf.summary.image('slerp', tf.expand_dims(slerp, 0))
        tf.summary.image('samples', tf.expand_dims(samples, 0))

        if FLAGS.dataset == 'lines32':
            batched = (n_interpolations, 32, n_images_per_interpolation, 32, 1)
            batched_interp = tf.transpose(
                tf.reshape(inter, batched), [0, 2, 1, 3, 4])
            mean_distance, mean_smoothness = tf.py_func(
                eval.line_eval, [batched_interp], [tf.float32, tf.float32])
            tf.summary.scalar('mean_distance', mean_distance)
            tf.summary.scalar('mean_smoothness', mean_smoothness)

        return ops


def main(argv):
    del argv  # Unused.
    batch = FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset, dict(batch_size=batch))
    scales = int(round(math.log(dataset.width // FLAGS.latent_width, 2)))
    print("scales: ", scales)
    model = MIDRAE(
        dataset,
        FLAGS.train_dir,
        latent=FLAGS.latent,
        depth=FLAGS.depth,
        scales=scales,
        advweight=FLAGS.advweight,
        advdepth=FLAGS.advdepth or FLAGS.depth,
        reg=FLAGS.reg,
        advnoise=FLAGS.advnoise,
        advfake=FLAGS.advfake,
        wgt_mmd=FLAGS.wgt_mmd)
    model.train()


if __name__ == '__main__':
    flags.DEFINE_integer('depth', 64, 'Depth of first for convolution.')
    flags.DEFINE_integer(
        'latent', 16,
        'Latent space depth, the total latent size is the depth multiplied by '
        'latent_width ** 2.')
    flags.DEFINE_integer('latent_width', 4, 'Width of the latent space.')
    flags.DEFINE_float('advweight', 0.5, 'Adversarial weight.')
    flags.DEFINE_integer('advdepth', 0, 'Depth for adversary network.')
    flags.DEFINE_float('reg', 0.2, 'Amount of discriminator regularization.')
    flags.DEFINE_float('wgt_mmd', 0.1, 'Weight of mmd')
    flags.DEFINE_float('advnoise', 0.1, 'weight of noise interpolation.')
    flags.DEFINE_float('advfake', 0.1, 'weight of fake interpolation.')
    app.run(main)
