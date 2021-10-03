import functools
import math
import types
from typing import Dict

import tensorflow as tf
from absl import flags, logging
from tensorflow.python.keras.utils import losses_utils

import simclr.resnet as resnet
from simclr.model import ProjectionHead, add_weight_decay
from simclr.resnet import BATCH_NORM_EPSILON

# import data_util


FLAGS = flags.FLAGS


class AtrousConv2D(tf.keras.layers.Layer):
    def __init__(self, depth, *args, **kwargs):
        super(AtrousConv2D, self).__init__(*args, **kwargs)
        self.depth = depth
        self.img_level_conv = tf.keras.layers.Conv2D(
            filters=depth,
            kernel_size=[1, 1],
            use_bias=False,
            name="image_level_conv_1x1",
        )
        self.at_pool1x1 = tf.keras.layers.Conv2D(
            filters=depth, kernel_size=[1, 1], name="conv_1x1_0"
        )
        self.bn_pool1x1 = tf.keras.layers.BatchNormalization(
            name="bn_pool1x1",
            epsilon=1e-5,
            momentum=0.997,
        )
        self.bn_pool3x3_1 = tf.keras.layers.BatchNormalization(
            name="bn_pool3x3_1",
            epsilon=1e-5,
            momentum=0.997,
        )
        self.bn_pool3x3_2 = tf.keras.layers.BatchNormalization(
            name="bn_pool3x3_2",
            epsilon=1e-5,
            momentum=0.997,
        )
        self.bn_pool3x3_3 = tf.keras.layers.BatchNormalization(
            name="bn_pool3x3_3",
            epsilon=1e-5,
            momentum=0.997,
        )
        self.at_pool3x3_1 = tf.keras.layers.Conv2D(
            filters=depth,
            kernel_size=[3, 3],
            padding="SAME",
            use_bias=False,
            name="conv_3x3_1",
            dilation_rate=6,
        )
        self.at_pool3x3_2 = tf.keras.layers.Conv2D(
            filters=depth,
            kernel_size=[3, 3],
            padding="SAME",
            use_bias=False,
            name="conv_3x3_2",
            dilation_rate=12,
        )
        self.at_pool3x3_3 = tf.keras.layers.Conv2D(
            filters=depth,
            kernel_size=[3, 3],
            padding="SAME",
            use_bias=False,
            name="conv_3x3_3",
            dilation_rate=18,
        )
        self.conv_output = tf.keras.layers.Conv2D(
            filters=depth, kernel_size=[1, 1], use_bias=False, name="conv_1x1_output"
        )
        self.bn_output = tf.keras.layers.BatchNormalization(
            name="bn_output",
            epsilon=1e-5,
            momentum=0.997,
        )

    def call(self, net):
        feature_map_size = tf.shape(net)
        # apply global average pooling
        image_level_features = tf.reduce_mean(
            net, [1, 2], name="image_level_global_pool", keepdims=True
        )
        image_level_features = self.img_level_conv(image_level_features)
        image_level_features = tf.image.resize(
            image_level_features, (feature_map_size[1], feature_map_size[2])
        )
        net1 = self.bn_pool1x1(self.at_pool1x1(net))
        net2 = self.bn_pool3x3_1(self.at_pool3x3_1(net))
        net3 = self.bn_pool3x3_2(self.at_pool3x3_2(net))
        net4 = self.bn_pool3x3_3(self.at_pool3x3_3(net))
        net = tf.concat(
            (image_level_features, net1, net2, net3, net4), axis=3, name="concat"
        )
        return self.bn_output(self.conv_output(net))


def build_atrous_conv_layer(
    x,
    depth,
    kernel_size,
    dilation_rate,
    prefix,
    padding="SAME",
    use_bn=False,
    depth_activation=False,
    global_bn=True,
    batch_norm_decay=0.9,
):
    x = tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=kernel_size,
        padding=padding,
        name=f"{prefix}_conv",
        dilation_rate=dilation_rate,
        use_bias=False,
    )(x)
    if use_bn:
        if global_bn:
            bn_cls = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_cls = tf.keras.layers.BatchNormalization
        x = bn_cls(
            momentum=batch_norm_decay, epsilon=BATCH_NORM_EPSILON, name=f"{prefix}_bn"
        )(x)

    if depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

    return x


def get_atrous_conv_logits(
    inputs,
    depth=256,
    padding="SAME",
    use_bn=False,
    global_bn=True,
    batch_norm_decay=0.9,
    depth_activation=False,
):
    # feature_map_size = tf.shape(inputs)
    # inp = tf.keras.Input(shape=inputs.shape[1:])
    inp = inputs
    # print(f"feature_map_size = {feature_map_size.numpy()}; inputs.shape = {inputs.shape}")
    # Global average image pooling - (b) in Fig. 5 of DeepLab v3 paper
    # TODO: if inputs is already 2D, then don't have to do reduce mean, since 2D input means that it's already gone through the projection head/ it's the average pool from resnet
    image_level_features = tf.reduce_mean(
        inp, [1, 2], name="image_level_global_pool", keepdims=True
    )
    image_level_features = tf.keras.layers.Conv2D(
        filters=depth, kernel_size=[1, 1], name="image_level_conv_1x1"
    )(image_level_features)
    image_level_features = tf.image.resize(
        image_level_features, tf.shape(inputs)[1:3], name="upsample_image_features"
    )

    # Atrous Spatial Pyramid Pooling - (a) in Fig. 5 of DeepLab v3 paper
    net1 = tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=[1, 1],
        use_bias=False,
        name="aspp0_1x1_conv",
        padding=padding,
    )(inp)
    if use_bn:
        if global_bn:
            bn = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn = tf.keras.layers.BatchNormalization
        net1 = bn(
            name="aspp0_1x1_bn", epsilon=BATCH_NORM_EPSILON, momentum=batch_norm_decay
        )(net1)

    net2 = build_atrous_conv_layer(
        inp,
        depth,
        3,
        6,
        "aspp1_3x3",
        padding=padding,
        use_bn=use_bn,
        depth_activation=depth_activation,
        global_bn=global_bn,
        batch_norm_decay=batch_norm_decay,
    )
    net3 = build_atrous_conv_layer(
        inp,
        depth,
        3,
        12,
        "aspp2_3x3",
        padding=padding,
        use_bn=use_bn,
        depth_activation=depth_activation,
        global_bn=global_bn,
        batch_norm_decay=batch_norm_decay,
    )
    net4 = build_atrous_conv_layer(
        inp,
        depth,
        3,
        18,
        "aspp3_3x3",
        padding=padding,
        use_bn=use_bn,
        depth_activation=depth_activation,
        global_bn=global_bn,
        batch_norm_decay=batch_norm_decay,
    )

    x = tf.concat(
        (image_level_features, net1, net2, net3, net4), axis=3, name="aspp_concat"
    )

    x = tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=[1, 1],
        padding=padding,
        name="conv_1x1_output",
        use_bias=False,
    )(x)
    if use_bn:
        if global_bn:
            bn = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn = tf.keras.layers.BatchNormalization
        x = bn(
            name="aspp_final_bn", epsilon=BATCH_NORM_EPSILON, momentum=batch_norm_decay
        )(x)
    if depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu, name="aspp_final_relu")(x)
    return x


class SegModel(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(self, config: Dict, **kwargs):
        super(SegModel, self).__init__(**kwargs)

        self.config = config
        self.train_mode = config["model_config"]["train_mode"]
        # Main model consists of resnet, proj head (and linear head if finetuning)
        with self.distribute_strategy.scope():  # TODO: see if can remove this
            self.resnet_model = resnet.resnet(config["model_config"])
            self._projection_head = ProjectionHead(config["model_config"])
            if self.train_mode == "finetune":
                self.atrous_block = AtrousConv2D(256)
                self.supervised_layer = tf.keras.layers.Conv2D(
                    filters=config["data_config"]["num_classes"],
                    kernel_size=1,
                    padding="SAME",
                    use_bias=False
                )

    def get_config(self):
        return self.config

    def call(self, inputs, training=False):
        # Base network forward pass.
        resnet_out = self.resnet_model(inputs, training=training, output_block=4)
        # Add heads.
        # TODO: Add segmentation projection head for contrastive loss based on pixel embedding
        # projection_head_outputs, supervised_head_inputs = self._projection_head(
        #     resnet_out, training
        # )
        supervised_head_inputs = projection_head_outputs = resnet_out
        if self.train_mode == "finetune":
            output = self.atrous_block(supervised_head_inputs)
            output = tf.image.resize(output, [self.image_size, self.image_size])
            supervised_output = self.supervised_layer(output, training=training)
            return resnet_out, None, supervised_output
        else:
            return resnet_out, projection_head_outputs, None

    def train_step(self, input_dict):
        loss = 0.0
        with tf.GradientTape() as tape:
            img = input_dict["image"]
            hiddens, projection_outputs, supervised_outputs = self(img, training=True)
            # if projection_outputs is not None:
            #     logits = projection_outputs
            # con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
            #     logits,
            #     hidden_norm=FLAGS.hidden_norm,
            #     temperature=FLAGS.temperature,
            #     strategy=strategy,
            # )
            # if loss is None:
            #     loss = con_loss
            # else:
            #     loss += con_loss
            if supervised_outputs is not None:  # will call this block in finetuning
                labels, logits = self.reshape_and_mask_labels_logits(
                    input_dict, supervised_outputs
                )
                sup_loss = self.update_loss_and_metrics(labels, logits)
                loss += sup_loss
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    def reshape_and_mask_labels_logits(self, input_dict, logits):
        labels = input_dict["seg_mask"]
        labels_reshaped = tf.reshape(labels, [-1, 1])
        logits_reshaped = tf.reshape(logits, (-1, self.num_classes))

        labels_reshaped, logits_reshaped = self.mask_out_labels_and_logits(
            input_dict, labels_reshaped, logits_reshaped
        )
        return labels_reshaped, logits_reshaped

    def mask_out_labels_and_logits(self, input_dict, labels_reshaped, logits_reshaped):
        if "image_mask" in input_dict:
            # mask out the labels:
            # True if include, False if masked out
            img_mask = input_dict["image_mask"]
            mask_reshaped = tf.reshape(img_mask, [-1, 1])
            labels_reshaped = tf.boolean_mask(labels_reshaped, mask_reshaped)
            logits_reshaped = tf.reshape(
                tf.boolean_mask(
                    logits_reshaped, tf.tile(mask_reshaped, [1, self.num_classes])
                ),
                [-1, self.num_classes],
            )
        return labels_reshaped, logits_reshaped

    def update_loss_and_metrics(self, labels_reshaped, logits_reshaped):
        # Update loss
        sup_loss = self.compiled_loss(labels_reshaped, logits_reshaped)

        # Update metrics
        if not self.compiled_metrics._built:
            self.compiled_metrics.build(logits_reshaped, labels_reshaped)

        # Recalculate X-entropy metric to make sure it matches with loss
        entropy_idx = self.metrics_names.index("crossentropy_loss")
        self.metrics[entropy_idx].update_state(labels_reshaped, logits_reshaped)

        predictions = tf.expand_dims(
            tf.argmax(logits_reshaped, axis=-1), axis=-1
        )  # n x 1
        predictions, labels_reshaped = losses_utils.squeeze_or_expand_dimensions(
            predictions, labels_reshaped
        )
        iou_idx = self.metrics_names.index("iou")
        acc_idx = self.metrics_names.index("accuracy")
        self.metrics[iou_idx].update_state(labels_reshaped, predictions)
        self.metrics[acc_idx].update_state(labels_reshaped, logits_reshaped)
        # Updates stateful loss metrics.
        self.compiled_loss(labels_reshaped, logits_reshaped)

        # weight_decay = add_weight_decay(
        #     self, self.weight_decay, self.optimizer_name, adjust_per_optimizer=True
        # )  # always 0 if self.weight_decay is 0

        # wt_decay_idx = self.metrics_names.index("weight_decay")
        # loss_mean_idx = self.metrics_names.index("total_loss_mean")
        # loss_sum_idx = self.metrics_names.index("total_loss_sum")

        # # update weight decay and total_loss metrics
        # self.metrics[wt_decay_idx].update_state(weight_decay)
        # self.metrics[loss_mean_idx].update_state(loss)
        # self.metrics[loss_sum_idx].update_state(loss)

        # The default behavior of `apply_gradients` is to sum gradients from all
        # replicas so we divide the loss by the number of replicas so that the
        # mean gradient is applied.
        # print(f"\n{loss} / {strategy.num_replicas_in_sync} = {loss/strategy.num_replicas_in_sync}\n")
        # loss = loss / strategy.num_replicas_in_sync
        return sup_loss

    def test_step(self, input_dict):
        """The logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.

        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.

        A `dict` containing values that will be passed to
        `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
        values of the `Model`'s metrics are returned.
        """
        img = input_dict["image"]
        _, _, supervised_outputs = self(img, training=True)
        if supervised_outputs is not None:  # will call this block in finetuning
            labels, logits = self.reshape_and_mask_labels_logits(
                input_dict, supervised_outputs
            )
            sup_loss = self.update_loss_and_metrics(labels, logits)

        return {m.name: m.result() for m in self.metrics}  # Return metrics

    def model(self, input_shape):
        # https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
        x = tf.keras.layers.Input(input_shape)
        return tf.keras.Model(inputs=[x], outputs=self(x))


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
