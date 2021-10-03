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

import lars_optimizer
import metrics
import objective as obj_lib
import resnet
import tensorflow as tf
from absl import flags

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
        config,
        name=None,
    ):
        super(WarmUpAndCosineDecay, self).__init__()
        self.parse_config(config)
        self._name = name

        iterations_per_epoch = self.num_examples // self.train_batch_size
        self.total_steps = self.num_epochs * iterations_per_epoch + 1
        self.calculate_scaled_lr()
        self.warmup_steps = int(round(self.warmup_epochs * iterations_per_epoch))
        with tf.name_scope(self._name or "WarmUpAndCosineDecay"):
            initial_learning_rate = float(self.scaled_lr)
            decay_steps = self.total_steps - self.warmup_steps
            self.cosine_decay = tf.keras.experimental.CosineDecay(
                initial_learning_rate, decay_steps
            )
        config["optimizer_config"]["learning_rate"] = self.scaled_lr

    def parse_config(self, config):
        opt_config = config["optimizer_config"]
        self.warmup_epochs = opt_config["warmup_epochs"]
        self.train_batch_size = config["data_config"]["batch_size"]
        self.num_examples = config["data_config"]["num_train"]
        self.num_epochs = config["train_config"]["num_epochs"]
        self.learning_rate_scaling = opt_config["lr_scaling"]
        self.base_learning_rate = opt_config["base_learning_rate"]

    def calculate_scaled_lr(self):
        if self.learning_rate_scaling == "linear":
            self.scaled_lr = self.base_learning_rate * self.train_batch_size / 256.0
        elif self.learning_rate_scaling == "sqrt":
            self.scaled_lr = self.base_learning_rate * math.sqrt(self.train_batch_size)
        else:
            raise ValueError(f"Unknown LR scaling {self.learning_rate_scaling}")

    def __call__(self, step):
        with tf.name_scope(self._name or "WarmUpAndCosineDecay"):
            if self.warmup_steps:
                learning_rate = tf.cast(step, tf.float32) / float(self.warmup_steps)
            else:
                learning_rate = self.scaled_lr

            # Cosine decay learning rate schedule
            learning_rate = tf.where(
                step < self.warmup_steps,
                learning_rate,
                self.cosine_decay(step - self.warmup_steps)
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
        model_config,
        **kwargs,
    ):
        self.num_layers = model_config["proj_config"]["num_proj_layers"]
        self.ft_proj_selector = model_config["proj_config"]["finetune_proj_selector"]
        self.head_mode = model_config["proj_config"]["proj_head_mode"]
        out_dim = model_config["proj_config"]["proj_out_dim"]
        self.linear_layers = []
        if self.head_mode == "none":
            pass  # directly use the output hiddens as hiddens
        elif self.head_mode == "linear":
            self.linear_layers = [
                LinearLayer(
                    num_classes=out_dim, use_bias=False, use_bn=True, name="l_0"
                )
            ]
            self.num_layers = 1
        elif self.head_mode == "nonlinear":
            for j in range(self.num_layers):
                is_middle_layer = j != self.num_layers - 1
                if is_middle_layer:  # use bias and relu for the output
                    use_bias = use_bn = True
                    num_classes = lambda input_shape: int(input_shape[-1])
                else:  # final layer
                    use_bias = False
                    use_bn = True
                    num_classes = out_dim
                self.linear_layers.append(
                    LinearLayer(
                        num_classes=num_classes,
                        use_bias=use_bias,
                        use_bn=use_bn,
                        name=f"nl_{j}",
                    )
                )
                if is_middle_layer:
                    self.linear_layers.append(
                        tf.keras.layers.ReLU()
                    )
        else:
            raise ValueError("Unknown head projection mode {}".format(self.head_mode))
        super(ProjectionHead, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs, training):
        if self.head_mode == "none":
            return inputs  # directly use the output hiddens as hiddens
        elif self.head_mode == "linear":
            assert len(self.linear_layers) == 1, len(self.linear_layers)

        hiddens_list = [tf.identity(inputs, "proj_head_input")]
        for j in range(self.num_layers):
            hiddens = self.linear_layers[j](hiddens_list[-1], training)
            hiddens_list.append(hiddens)
        # The first element is the output of the projection head.
        # The second element is the input of the finetune head.
        proj_head_output = tf.identity(hiddens_list[-1], "proj_head_output")
        return proj_head_output, hiddens_list[self.ft_proj_selector]


class SupervisedHead(tf.keras.layers.Layer):
    def __init__(self, num_classes, name="head_supervised", **kwargs):
        super(SupervisedHead, self).__init__(name=name, **kwargs)
        self.linear_layer = LinearLayer(num_classes)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_classes": self.linear_layer
        })
        return config

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
        weight_decay=0,
        resnet_depth=50,
        sk_ratio=0.0,
        width_multiplier=1,
        proj_out_dim=128,
        num_proj_layers=3,
        ft_proj_selector=0,
        head_mode="nonlinear",
        fine_tune_after_block=-1,
        linear_eval_while_pretraining=False,
        weighted_loss=False,
        **kwargs,
    ):
        super(Model, self).__init__(**kwargs)

        ## keep track of all kwargs
        self.num_classes = num_classes
        self.image_size = image_size
        self.train_mode = train_mode
        self.fine_tune_after_block = fine_tune_after_block
        self.linear_eval_while_pretraining = linear_eval_while_pretraining
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.resnet_depth = resnet_depth
        self.sk_ratio = sk_ratio
        self.width_multiplier = width_multiplier
        self.proj_out_dim = proj_out_dim
        self.num_proj_layers = num_proj_layers
        self.ft_proj_selector = ft_proj_selector
        self.head_mode = head_mode
        self.fine_tune_after_block = fine_tune_after_block
        self.linear_eval_while_pretraining = linear_eval_while_pretraining
        self.weighted_loss = weighted_loss
        # Main model consists of resnet, proj head (and linear head if finetuning)
        with self.distribute_strategy.scope():
            self.resnet_model = resnet.resnet(
                resnet_depth=resnet_depth,
                width_multiplier=width_multiplier,
                train_mode=train_mode,
                fine_tune_after_block=fine_tune_after_block,
                cifar_stem=image_size <= 32,
                sk_ratio=sk_ratio,
            )
            self._projection_head = ProjectionHead(
                proj_out_dim,
                num_proj_layers,
                ft_proj_selector,
                head_mode,
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

    def get_config(self):
        config = {}
        config.update(
            {
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "train_mode": self.train_mode,
                "fine_tune_after_block": self.fine_tune_after_block,
                "linear_eval_while_pretraining": self.linear_eval_while_pretraining,
                "optimizer_name": self.optimizer_name,
                "weight_decay": self.weight_decay,
                "resnet_depth": self.resnet_depth,
                "sk_ratio": self.sk_ratio,
                "width_multiplier": self.width_multiplier,
                "proj_out_dim": self.proj_out_dim,
                "num_proj_layers": self.num_proj_layers,
                "ft_proj_selector": self.ft_proj_selector,
                "head_mode": self.head_mode,
                "fine_tune_after_block": self.fine_tune_after_block,
                "linear_eval_while_pretraining": self.linear_eval_while_pretraining,
            }
        )
        return config

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
        if not self.weighted_loss:
            #! THere is a bug here
            img = img_labels[0]
            labels = img_labels[1]
            weights = None
        else:
            img, labels = img_labels[0]
            weights = img_labels[1]
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
                # compiled_loss = sparse categorical x-entropy w/ logits
                sup_loss = self.compiled_loss(tf.squeeze(labels), logits)
                if self.weighted_loss:
                    weighted_loss = sup_loss * weights
                    sup_loss = tf.reduce_mean(weighted_loss)
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
            loss = tf.cast(loss, tf.float32)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    def model(self, input_shape):
        # https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
        x = tf.keras.layers.Input(input_shape)
        return tf.keras.Model(inputs=[x], outputs=self(x))
