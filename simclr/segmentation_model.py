import functools
import math

# import data_util
import lars_optimizer
import metrics
import objective as obj_lib
import resnet
import tensorflow as tf
from absl import flags, logging
from model import ProjectionHead, add_weight_decay
from resnet import BATCH_NORM_EPSILON

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


def AtrousConv_BN(
    x,
    filters,
    prefix,
    stride=1,
    kernel_size=3,
    rate=1,
    depth_activation=False,
    center=True,
    scale=True,
    # epsilon=1e-3,
    global_bn=True,
    batch_norm_decay=0.9,
):
    """Atrous convolutional layer with BN between depthwise & pointwise. Optionally add activation after BN
    Implements right "same" padding for even kernel sizes
    Args:
        x: input tensor
        filters: num of filters in pointwise convolution
        prefix: prefix before name
        stride: stride at depthwise conv
        kernel_size: kernel size for depthwise convolution
        rate: atrous rate for depthwise convolution
        depth_activation: flag to use activation between depthwise & poinwise convs
        epsilon: epsilon to use in BN layer
    Note: Taken from https://github.com/bonlime/keras-deeplab-v3-plus/blob/e32f66b21a9647dfe08f27549e3200526e959ab1/model.py#L48
    """

    bn_axis = -1  # data_format == 'channels_last'
    gamma_initializer = tf.ones_initializer()
    if global_bn:
        bn_cls = functools.partial(
            tf.keras.layers.experimental.SyncBatchNormalization,
            axis=bn_axis,
            momentum=batch_norm_decay,
            epsilon=BATCH_NORM_EPSILON,
            center=center,
            scale=scale,
            gamma_initializer=gamma_initializer,  # assumes init_zero == False
        )
    else:
        bn_cls = functools.partial(
            tf.keras.layers.BatchNormalization,
            axis=bn_axis,
            momentum=batch_norm_decay,
            epsilon=BATCH_NORM_EPSILON,
            center=center,
            scale=scale,
            fused=False,
            gamma_initializer=gamma_initializer,
        )

    if stride == 1:
        depth_padding = "same"
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = "valid"

    if not depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

    # Depthwise Conv, BN, and optional relu
    x = tf.keras.layers.DepthwiseConv2D(
        (kernel_size, kernel_size),
        strides=(stride, stride),
        dilation_rate=(rate, rate),
        padding=depth_padding,
        use_bias=False,
        name=prefix + "_depthwise",
    )(x)
    x = bn_cls(name=prefix + "_depthwise_BN")(x)
    if depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

    # Atrous Conv2D
    x = tf.keras.layers.Conv2D(
        filters, (1, 1), padding="same", use_bias=False, name=prefix + "_pointwise"
    )(x)
    x = bn_cls(name=prefix + "_pointwise_BN")(x)
    if depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

    return x


def AtrousConvBlock(
    x,
    depth=256,
    padding="SAME",
    use_bn=False,
    global_bn=True,
    batch_norm_decay=0.9,
    depth_activation=False,
):
    inp = tf.keras.layers.Input(shape=x.shape)
    output = get_aspp_logits(
        x,
        depth=depth,
        padding=padding,
        use_bn=use_bn,
        global_bn=global_bn,
        batch_norm_decay=batch_norm_decay,
        depth_activation=depth_activation,
    )

    return tf.keras.models.Model(inputs=tf.keras.layers.Input(x), outputs=output)


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
        **kwargs,
    ):
        super(SegModel, self).__init__(**kwargs)

        # keep track of all kwargs
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
        # Main model consists of resnet, proj head (and linear head if finetuning)
        with self.distribute_strategy.scope():
            self.resnet_model = resnet.resnet(
                resnet_depth=resnet_depth,
                width_multiplier=width_multiplier,
                train_mode=train_mode,
                fine_tune_after_block=fine_tune_after_block,
                sk_ratio=sk_ratio,
            )
            self._projection_head = ProjectionHead(
                proj_out_dim,
                num_proj_layers,
                ft_proj_selector,
                head_mode,
            )
            if train_mode == "finetune":
                dummy_inp = tf.keras.Input((self.image_size, self.image_size, 3))
                self.atrous_block = AtrousConv2D(256)
                self.supervised_layer = tf.keras.layers.Conv2D(
                    filters=num_classes, kernel_size=1, padding="SAME", use_bias=False
                )

    def get_config(self):
        # config = super().get_config().copy()
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
        hiddens = self.resnet_model(features, training=training, output_block=4)
        # Add heads.
        # TODO: Add segmentation projection head for contrastive loss based on pixel embedding
        # projection_head_outputs, supervised_head_inputs = self._projection_head(
        #     hiddens, training
        # )
        supervised_head_inputs = hiddens
        projection_head_outputs = hiddens
        if self.train_mode == "finetune":
            output = self.atrous_block(supervised_head_inputs)
            output = tf.image.resize(output, [self.image_size, self.image_size])
            supervised_output = self.supervised_layer(output, training=training)
            return None, supervised_output
        else:
            return projection_head_outputs, None

    def train_step(self, img_labels):
        img = img_labels[0]
        labels = img_labels[1]
        strategy = self.distribute_strategy
        with tf.GradientTape() as tape:
            projection_outputs, supervised_outputs = self(img, training=True)

            loss = None
            # if projection_outputs is not None:
            #     logits = projection_outputs
            #     con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
            #         logits,
            #         hidden_norm=FLAGS.hidden_norm,
            #         temperature=FLAGS.temperature,
            #         strategy=strategy,
            #     )
            #     if loss is None:
            #         loss = con_loss
            #     else:
            #         loss += con_loss
            if supervised_outputs is not None:  # will call this block in finetuning
                logits = supervised_outputs
                # num_rows = tf.math.reduce_prod(tf.shape(img)[0:-1])
                labels_reshaped = tf.reshape(labels, [-1, 1])
                logits_reshaped = tf.reshape(logits, (-1, self.num_classes))
                if not self.compiled_metrics._built:
                    self.compiled_metrics.build(logits_reshaped, labels_reshaped)
                sup_loss = self.compiled_loss(labels_reshaped, logits_reshaped)
                # sup_loss_idx = self.metrics_names.index("supervised_loss_mean")
                # self.metrics[sup_loss_idx].update_state(sup_loss)

                # Recalculate X-entropy metric to make sure it matches with loss
                entropy_idx = self.metrics_names.index("crossentropy_loss")
                self.metrics[entropy_idx].update_state(labels_reshaped, logits_reshaped)
                if loss is None:
                    loss = sup_loss
                else:
                    loss += sup_loss
                iou_idx = self.metrics_names.index("iou")
                acc_idx = self.metrics_names.index("accuracy")
                predictions = tf.expand_dims(
                    tf.argmax(logits_reshaped, axis=-1), axis=-1
                )  # n x 1
                self.metrics[iou_idx].update_state(labels_reshaped, predictions)
                self.metrics[acc_idx].update_state(labels_reshaped, logits_reshaped)

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

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, img_labels):
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

        Args:
            data: A nested structure of `Tensor`s.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
            values of the `Model`'s metrics are returned.
        """
        img = img_labels[0]
        labels = img_labels[1]
        _, supervised_outputs = self(img, training=True)
        logits = supervised_outputs
        # num_rows = tf.math.reduce_prod(tf.shape(img)[0:-1])
        labels_reshaped = tf.reshape(labels, [-1, 1])
        logits_reshaped = tf.reshape(logits, (-1, self.num_classes))
        if not self.compiled_metrics._built:
            self.compiled_metrics.build(logits_reshaped, labels_reshaped)
        sup_loss = self.compiled_loss(labels_reshaped, logits_reshaped)
        # sup_loss_idx = self.metrics_names.index("supervised_loss_mean")
        # self.metrics[sup_loss_idx].update_state(sup_loss)

        # Recalculate X-entropy metric to make sure it matches with loss
        entropy_idx = self.metrics_names.index("crossentropy_loss")
        self.metrics[entropy_idx].update_state(labels_reshaped, logits_reshaped)
        iou_idx = self.metrics_names.index("iou")
        acc_idx = self.metrics_names.index("accuracy")
        predictions = tf.expand_dims(
            tf.argmax(logits_reshaped, axis=-1), axis=-1
        )  # n x 1
        self.metrics[iou_idx].update_state(labels_reshaped, predictions)
        self.metrics[acc_idx].update_state(labels_reshaped, logits_reshaped)
        # Updates stateful loss metrics.
        self.compiled_loss(labels_reshaped, logits_reshaped)
        # Return metrics
        return {m.name: m.result() for m in self.metrics}

    def model(self, input_shape):
        # https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
        x = tf.keras.layers.Input(input_shape)
        return tf.keras.Model(inputs=[x], outputs=self(x))
