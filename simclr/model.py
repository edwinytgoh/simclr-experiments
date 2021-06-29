# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Model specification for SimCLR."""

import math
from absl import flags

import data_util
import lars_optimizer
import resnet
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

from config import num_classes, train_mode, image_size

def build_optimizer(learning_rate):
    """Returns the optimizer."""
    if FLAGS.optimizer == 'momentum':
        return tf.keras.optimizers.SGD(learning_rate, FLAGS.momentum, nesterov=True)
    elif FLAGS.optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate)
    elif FLAGS.optimizer == 'lars':
        return lars_optimizer.LARSOptimizer(
                learning_rate,
                momentum=FLAGS.momentum,
                weight_decay=FLAGS.weight_decay,
                exclude_from_weight_decay=[
                        'batch_normalization', 'bias', 'head_supervised'
                ])
    else:
        raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))


def add_weight_decay(model, adjust_per_optimizer=True):
    """Compute weight decay from flags."""
    if adjust_per_optimizer and 'lars' in FLAGS.optimizer:
        # Weight decay are taking care of by optimizer for these cases.
        # Except for supervised head, which will be added here.
        l2_losses = [
                tf.nn.l2_loss(v)
                for v in model.trainable_variables
                if 'head_supervised' in v.name and 'bias' not in v.name
        ]
        if l2_losses:
            return FLAGS.weight_decay * tf.add_n(l2_losses)
        else:
            return 0

    # TODO(srbs): Think of a way to avoid name-based filtering here.
    l2_losses = [
            tf.nn.l2_loss(v)
            for v in model.trainable_weights
            if 'batch_normalization' not in v.name
    ]
    loss = FLAGS.weight_decay * tf.add_n(l2_losses)
    return loss


def get_train_steps(num_examples):
    """Determine the number of training steps."""
    return FLAGS.train_steps or (
            num_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1)


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, base_learning_rate, num_examples, name=None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            warmup_steps = int(
                    round(FLAGS.warmup_epochs * self.num_examples //
                                FLAGS.train_batch_size))
            if FLAGS.learning_rate_scaling == 'linear':
                scaled_lr = self.base_learning_rate * FLAGS.train_batch_size / 256.
            elif FLAGS.learning_rate_scaling == 'sqrt':
                scaled_lr = self.base_learning_rate * math.sqrt(FLAGS.train_batch_size)
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                        FLAGS.learning_rate_scaling))
            learning_rate = (
                    step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

            # Cosine decay learning rate schedule
            total_steps = get_train_steps(self.num_examples)
            # TODO(srbs): Cache this object.
            cosine_decay = tf.keras.experimental.CosineDecay(
                    scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                                             cosine_decay(step - warmup_steps))

            return learning_rate

    def get_config(self):
        return {
                'base_learning_rate': self.base_learning_rate,
                'num_examples': self.num_examples,
        }


class LinearLayer(tf.keras.layers.Layer):

    def __init__(self,
                             num_classes,
                             use_bias=True,
                             use_bn=False,
                             name='linear_layer',
                             **kwargs):
        # Note: use_bias is ignored for the dense layer when use_bn=True.
        # However, it is still used for batch norm.
        super(LinearLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.use_bn = use_bn
        self._name = name
        if callable(self.num_classes):
            num_classes = -1
        else:
            num_classes = self.num_classes
        self.dense = tf.keras.layers.Dense(
                num_classes,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                use_bias=use_bias and not self.use_bn)
        if self.use_bn:
            self.bn_relu = resnet.BatchNormRelu(relu=False, center=use_bias)

    def build(self, input_shape):
        # TODO(srbs): Add a new SquareDense layer.
        if callable(self.num_classes):
            self.dense.units = self.num_classes(input_shape)
        super(LinearLayer, self).build(input_shape)

    def call(self, inputs, training):
        assert inputs.shape.ndims == 2, inputs.shape
        inputs = self.dense(inputs)
        if self.use_bn:
            inputs = self.bn_relu(inputs, training=training)
        return inputs


class ProjectionHead(tf.keras.layers.Layer):

    def __init__(self, out_dim, num_layers, ft_proj_selector, head_mode, use_bias, use_bn, **kwargs):
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.ft_proj_selector = ft_proj_selector  # for finetuning
        self.linear_layers = []
        self.head_mode = head_mode
        if self.head_mode == 'none':
            pass    # directly use the output hiddens as hiddens
        elif self.head_mode == 'linear':
            self.linear_layers = [
                LinearLayer(
                    num_classes=out_dim, use_bias=use_bias, use_bn=use_bn, name='l_0')
            ]
        elif self.head_mode == 'nonlinear':
            for j in range(num_layers):
                if j != num_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    self.linear_layers.append(
                            LinearLayer(
                                    num_classes=lambda input_shape: int(input_shape[-1]),
                                    use_bias=True,
                                    use_bn=True,
                                    name='nl_%d' % j))
                else:
                    # for the final layer, neither bias nor relu is used.
                    self.linear_layers.append(
                            LinearLayer(
                                    num_classes=out_dim,
                                    use_bias=False,
                                    use_bn=True,
                                    name='nl_%d' % j))
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                    head_mode))
        super(ProjectionHead, self).__init__(**kwargs)

    def call(self, inputs, training):
        if self.head_mode == 'none':
            return inputs    # directly use the output hiddens as hiddens
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]
        if self.head_mode == 'linear':
            assert len(self.linear_layers) == 1, len(self.linear_layers)
            return hiddens_list.append(self.linear_layers[0](hiddens_list[-1], training))
        elif self.head_mode == 'nonlinear':
            for j in range(self.num_layers):
                hiddens = self.linear_layers[j](hiddens_list[-1], training)
                if j != self.num_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    hiddens = tf.nn.relu(hiddens)
                hiddens_list.append(hiddens)
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                    self.head_mode))
        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
        return proj_head_output, hiddens_list[self.ft_proj_selector]


class SupervisedHead(tf.keras.layers.Layer):

    def __init__(self, num_classes, name='head_supervised', **kwargs):
        super(SupervisedHead, self).__init__(name=name, **kwargs)
        self.linear_layer = LinearLayer(num_classes)

    def call(self, inputs, training):
        inputs = self.linear_layer(inputs, training)
        inputs = tf.identity(inputs, name='logits_sup')
        return inputs


class Model(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(
        self,
        num_classes,
        image_size,
        train_mode='finetune',
        resnet_depth=50,
        width_multiplier=1,
        proj_out_dim=128,
        num_proj_layers=3,
        ft_proj_selector=0,
        head_mode='linear',
        use_bias=False,  # whether to use bias in projection head
        use_bn=True,  # whether to use batch norm in projection head
        finetune_after_block=-1,
        **kwargs):
        super(Model, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.image_size = image_size
        self.train_mode = train_mode
        self.finetune_after_block = finetune_after_block
        self.resnet_model = resnet.resnet(
                resnet_depth=resnet_depth,
                width_multiplier=width_multiplier,
                cifar_stem=image_size <= 32)
        self._projection_head = ProjectionHead(
            proj_out_dim, num_proj_layers, ft_proj_selector, head_mode, use_bias, use_bn
        )
        if train_mode == 'finetune':
            self.supervised_head = SupervisedHead(num_classes)

    def __call__(self, inputs, training):
        features = inputs

        # Base network forward pass.
        hiddens = self.resnet_model(features, training=training)

        # Add heads.
        projection_head_outputs, supervised_head_inputs = self._projection_head(
                hiddens, training)

        if self.train_mode == 'finetune':
            supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                                                                         training)
            return None, supervised_head_outputs
        elif self.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                    tf.stop_gradient(supervised_head_inputs), training)
            return projection_head_outputs, supervised_head_outputs
        else:
            return projection_head_outputs, None

    def train_step(self, img_labels):
        if self.train_mode == 'pretrain':
            num_transforms = 2
            if self.fine_tune_after_block > -1:
                raise ValueError('Does not support layer freezing during pretraining,'
                                                 'should set fine_tune_after_block<=-1 for safety.')
        else:
            num_transforms = 1

        # Split channels, and optionally apply extra batched augmentation.
        features_list = tf.split(
                features, num_or_size_splits=num_transforms, axis=-1)
        if FLAGS.use_blur and training and self.train_mode == 'pretrain':
            features_list = data_util.batch_random_blur(features_list, self.image_size, self.image_size)
        features = tf.concat(features_list, 0)    # (num_transforms * bsz, h, w, c)


        # see: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        img = img_labels[0]
        labels = img_labels[1]
        with tf.GradientTape() as tape:
            logits = self(img)  # forward pass
            # loss function is configured in compile()
            loss = self.compiled_loss(tf.squeeze(labels), logits, regularization_losses=self.losses)

        # compute gradients and update weights
        dense_layer_weights = self.dense_layer.trainable_weights
        grads = tape.gradient(loss, dense_layer_weights)
        self.optimizer.apply_gradients(zip(grads, dense_layer_weights))

        # update metrics
        self.compiled_metrics.update_state(labels, logits)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def test_model_lib():
    global train_mode, image_size, num_classes
    print(
        f"num_classes = {num_classes}",
        f"image_size = {image_size}"
    )