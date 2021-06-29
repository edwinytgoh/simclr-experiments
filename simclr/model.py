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
import tensorflow as tf

import objective as obj_lib
import metrics


FLAGS = flags.FLAGS


def build_optimizer(learning_rate, optimizer, momentum, weight_decay=1e-6):
    """Returns the optimizer."""
    if optimizer == "momentum":
        return tf.keras.optimizers.SGD(learning_rate, momentum, nesterov=True)
    elif optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate)
    elif optimizer == "lars":
        return lars_optimizer.LARSOptimizer(
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            exclude_from_weight_decay=[
                "batch_normalization",
                "bias",
                "head_supervised",
            ],
        )
    else:
        raise ValueError("Unknown optimizer {}".format(FLAGS.optimizer))


def add_weight_decay(model, weight_decay, optimizer, adjust_per_optimizer=True):
    """Compute weight decay from flags."""
    if adjust_per_optimizer and "lars" in optimizer:
        # Weight decay are taking care of by optimizer for these cases.
        # Except for supervised head, which will be added here.
        l2_losses = [
            tf.nn.l2_loss(v)
            for v in model.trainable_variables
            if "head_supervised" in v.name and "bias" not in v.name
        ]
        if l2_losses:
            return weight_decay * tf.add_n(l2_losses)
        else:
            return 0

    # TODO(srbs): Think of a way to avoid name-based filtering here.
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_weights
        if "batch_normalization" not in v.name
    ]
    loss = weight_decay * tf.add_n(l2_losses)
    return loss


def get_train_steps(num_examples):
    """Determine the number of training steps."""
    return FLAGS.train_steps or (
        num_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1
    )


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(
        self,
        base_learning_rate,
        num_examples,
        warmup_epochs,
        total_epochs,
        train_batch_size,
        learning_rate_scaling,
        name=None,
    ):
        super(WarmUpAndCosineDecay, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.train_batch_size = train_batch_size
        self.total_steps = num_examples * total_epochs // train_batch_size + 1
        self.learning_rate_scaling = learning_rate_scaling
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples

        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or "WarmUpAndCosineDecay"):
            warmup_steps = int(
                round(self.warmup_epochs * self.num_examples // self.train_batch_size)
            )
            if self.learning_rate_scaling == "linear":
                scaled_lr = self.base_learning_rate * self.train_batch_size / 256.0
            elif self.learning_rate_scaling == "sqrt":
                scaled_lr = self.base_learning_rate * math.sqrt(self.train_batch_size)
            else:
                raise ValueError(
                    "Unknown learning rate scaling {}".format(
                        self.learning_rate_scaling
                    )
                )
            learning_rate = (
                step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr
            )

            # Cosine decay learning rate schedule
            # TODO(srbs): Cache this object.
            cosine_decay = tf.keras.experimental.CosineDecay(
                scaled_lr, self.total_steps - warmup_steps
            )
            learning_rate = tf.where(
                step < warmup_steps, learning_rate, cosine_decay(step - warmup_steps)
            )

            return learning_rate

    def get_config(self):
        return {
            "base_learning_rate": self.base_learning_rate,
            "num_examples": self.num_examples,
        }


class LinearLayer(tf.keras.layers.Layer):
    def __init__(
        self, num_classes, use_bias=True, use_bn=False, name="linear_layer", **kwargs
    ):
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
            use_bias=use_bias and not self.use_bn,
        )
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
    def __init__(
        self,
        out_dim,
        num_layers,
        ft_proj_selector,
        head_mode,
        use_bias,
        use_bn,
        **kwargs,
    ):
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.ft_proj_selector = ft_proj_selector  # for finetuning
        self.linear_layers = []
        self.head_mode = head_mode
        if self.head_mode == "none":
            pass  # directly use the output hiddens as hiddens
        elif self.head_mode == "linear":
            self.linear_layers = [
                LinearLayer(
                    num_classes=out_dim, use_bias=use_bias, use_bn=use_bn, name="l_0"
                )
            ]
        elif self.head_mode == "nonlinear":
            for j in range(num_layers):
                if j != num_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=lambda input_shape: int(input_shape[-1]),
                            use_bias=True,
                            use_bn=True,
                            name="nl_%d" % j,
                        )
                    )
                else:
                    # for the final layer, neither bias nor relu is used.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=out_dim,
                            use_bias=False,
                            use_bn=True,
                            name="nl_%d" % j,
                        )
                    )
        else:
            raise ValueError("Unknown head projection mode {}".format(head_mode))
        super(ProjectionHead, self).__init__(**kwargs)

    def call(self, inputs, training):
        if self.head_mode == "none":
            return inputs  # directly use the output hiddens as hiddens
        hiddens_list = [tf.identity(inputs, "proj_head_input")]
        if self.head_mode == "linear":
            assert len(self.linear_layers) == 1, len(self.linear_layers)
            hiddens_list.append(self.linear_layers[0](hiddens_list[-1], training))
            return hiddens_list
        elif self.head_mode == "nonlinear":
            for j in range(self.num_layers):
                hiddens = self.linear_layers[j](hiddens_list[-1], training)
                if j != self.num_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    hiddens = tf.nn.relu(hiddens)
                hiddens_list.append(hiddens)
        else:
            raise ValueError("Unknown head projection mode {}".format(self.head_mode))
        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], "proj_head_output")
        return proj_head_output, hiddens_list[self.ft_proj_selector]


class SupervisedHead(tf.keras.layers.Layer):
    def __init__(self, num_classes, name="head_supervised", **kwargs):
        super(SupervisedHead, self).__init__(name=name, **kwargs)
        self.linear_layer = LinearLayer(num_classes)

    def call(self, inputs, training):
        inputs = self.linear_layer(inputs, training)
        inputs = tf.identity(inputs, name="logits_sup")
        return inputs


class Model(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(
        self,
        num_classes,
        image_size,
        train_mode,
        optimizer_name,
        weight_decay,
        resnet_depth=50,
        sk_ratio=0.0,
        width_multiplier=1,
        proj_out_dim=128,
        num_proj_layers=3,
        ft_proj_selector=0,
        head_mode="linear",
        use_bias=False,  # whether to use bias in projection head
        use_bn=True,  # whether to use batch norm in projection head
        finetune_after_block=-1,
        linear_eval_while_pretraining=False,
        **kwargs,
    ):
        super(Model, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.image_size = image_size
        self.train_mode = train_mode
        self.finetune_after_block = finetune_after_block
        self.linear_eval_while_pretraining = linear_eval_while_pretraining
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        # Main model consists of resnet, proj head (and linear head if finetuning)
        with self.distribute_strategy.scope():
            self.resnet_model = resnet.resnet(
                resnet_depth=resnet_depth,
                width_multiplier=width_multiplier,
                cifar_stem=image_size <= 32,
                sk_ratio=sk_ratio,
            )
            self._projection_head = ProjectionHead(
                proj_out_dim,
                num_proj_layers,
                ft_proj_selector,
                head_mode,
                use_bias,
                use_bn,
            )
            if train_mode == "finetune":
                self.supervised_head = SupervisedHead(num_classes)

        # if train_mode == "pretrain":
        #     self.contrast_loss_metric = tf.keras.metrics.Mean("train/contrast_loss")
        #     self.contrast_acc_metric = tf.keras.metrics.Mean("train/contrast_acc")
        #     self.contrast_entropy_metric = tf.keras.metrics.Mean(
        #         "train/contrast_entropy"
        #     )
        #     self.all_metrics.extend(
        #         [
        #             self.contrast_loss_metric,
        #             self.contrast_acc_metric,
        #             self.contrast_entropy_metric,
        #         ]
        #     )

        # self.weight_decay_metric

    def call(self, inputs, training=False):
        features = inputs

        # Base network forward pass.
        hiddens = self.resnet_model(features, training=training)
        # Add heads.
        projection_head_outputs, supervised_head_inputs = self._projection_head(
            hiddens, training
        )
        if self.train_mode == "finetune":
            supervised_head_outputs = self.supervised_head(
                supervised_head_inputs, training
            )
            return None, supervised_head_outputs
        elif self.train_mode == "pretrain" and self.lineareval_while_pretraining:
            # When performing pretraining and linear evaluation together we do not
            # want information from linear eval flowing back into pretraining network
            # so we put a stop_gradient.
            supervised_head_outputs = self.supervised_head(
                tf.stop_gradient(supervised_head_inputs), training
            )
            return projection_head_outputs, supervised_head_outputs
        else:
            return projection_head_outputs, None

    def train_step(self, img_labels):
        img = img_labels[0]
        labels = img_labels[1]
        strategy = self.distribute_strategy
        with tf.GradientTape() as tape:
            projection_outputs, supervised_outputs = self(img, training=True)
            loss = None
            if projection_outputs is not None:
                logits = projection_outputs
                con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
                    logits,
                    hidden_norm=FLAGS.hidden_norm,
                    temperature=FLAGS.temperature,
                    strategy=strategy,
                )
                if loss is None:
                    loss = con_loss
                else:
                    loss += con_loss
                # metrics.update_pretrain_metrics_train(
                #     self.metrics_dict["contrastive_loss"],
                #     con_loss,
                #     logits_con,
                #     labels_con,
                # )
            if supervised_outputs is not None:  # will call this block in finetuning
                logits = supervised_outputs
                sup_loss = self.compiled_loss(tf.squeeze(labels), logits)
                # l = labels
                # if (
                #     FLAGS.train_mode == "pretrain"
                #     and FLAGS.lineareval_while_pretraining
                # ):
                #     l = tf.concat([l, l], 0)
                #     sup_loss = obj_lib.add_supervised_loss(labels=l, logits=outputs)
                if loss is None:
                    loss = sup_loss
                else:
                    loss += sup_loss

            if not self.compiled_metrics._built:
                self.compiled_metrics.build(logits, labels)

            supervised_loss_metric = self.metrics[
                self.metrics_names.index("train/supervised_loss")
            ]
            supervised_acc_metric = self.metrics[
                self.metrics_names.index("train/supervised_acc")
            ]
            metrics.update_finetune_metrics_train(
                supervised_loss_metric, supervised_acc_metric, sup_loss, labels, logits
            )
            weight_decay = add_weight_decay(
                self, self.weight_decay, self.optimizer_name, adjust_per_optimizer=True
            )
            self.metrics[self.metrics_names.index("train/weight_decay")].update_state(
                weight_decay
            )
            loss += weight_decay
            self.metrics[self.metrics_names.index("train/total_loss")].update_state(
                loss
            )  # update total_loss metric
            # The default behavior of `apply_gradients` is to sum gradients from all
            # replicas so we divide the loss by the number of replicas so that the
            # mean gradient is applied.
            loss = loss / strategy.num_replicas_in_sync
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    def model(self, input_shape):
        # https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
        x = tf.keras.layers.Input(input_shape)
        return tf.keras.Model(inputs=[x], outputs=self(x))
