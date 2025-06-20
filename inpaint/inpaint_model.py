import tensorflow as tf
from .inpaint_convolution import standard_convolution_layer, standard_deconvolution_layer
from .region_convoluiton import region_convolution, region_deconvolution


def region_wise_generator(x, mask, padding="SAME", name="inpaint_net", reuse=False):
    """
    Region-wise generator
    Args:
            x: incomplete image
            mask: mask region {0,1}
    returns:
            predicted image
    """

    x1 = semantic_inferring_network(x, mask, reuse=reuse)
    x_combine = x * mask + x1 * (1 - mask)
    x2 = global_perceiving_network(x_combine, mask, reuse=reuse)
    return x2


def semantic_inferring_network(x, mask, padding="SAME", name="inpaint_net", reuse=False):
    """
    Semantic inferring network
    Args:
            x: incomplete image
            mask: mask region {0,1}
    returns:
            image predicted by semantic inferring network
    """
    channel_number = 32
    tf.ones_like(x)[:, :, :, 0:1]
    x = tf.concat([x, mask], axis=3)
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        x1 = standard_convolution_layer(x, mask, channel_number, 5, 1, name="conv1")
        x2 = standard_convolution_layer(x1, mask, 2 * channel_number, 3, 2, name="conv2_downsample")
        x3 = standard_convolution_layer(x2, mask, 2 * channel_number, 3, 1, name="conv3")
        x4 = standard_convolution_layer(x3, mask, 4 * channel_number, 3, 2, name="conv4_downsample")
        x5 = standard_convolution_layer(x4, mask, 4 * channel_number, 3, 1, name="conv5")
        x6 = standard_convolution_layer(x5, mask, 4 * channel_number, 3, 1, name="conv6")

        # dilated conv
        x7 = standard_convolution_layer(x6, mask, 4 * channel_number, 3, dilation_rate=2, name="conv7_atrous")
        x8 = standard_convolution_layer(x7, mask, 4 * channel_number, 3, dilation_rate=4, name="conv8_atrous")
        x9 = standard_convolution_layer(x8, mask, 4 * channel_number, 3, dilation_rate=8, name="conv9_atrous")
        x10 = standard_convolution_layer(x9, mask, 4 * channel_number, 3, dilation_rate=16, name="conv10_atrous")

        x11 = standard_convolution_layer(
            tf.concat([x10, x6], axis=-1), mask, 4 * channel_number, 3, 1, name="conv11"
        )
        x12 = standard_convolution_layer(
            tf.concat([x11, x5], axis=-1), mask, 4 * channel_number, 3, 1, name="conv12"
        )

        x_complete, x_missing = tf.concat([x12, x4], axis=-1), x12
        x13 = region_deconvolution(x_complete, x_missing, mask, layer_name="com_13")

        x_complete, x_missing = tf.concat([x13, x3], axis=-1), x13
        x14 = region_convolution(x_complete, x_missing, mask, layer_name="com_14")

        x_complete, x_missing = tf.concat([x14, x2], axis=-1), x14
        x15 = region_deconvolution(x_complete, x_missing, mask, layer_name="com_15")

        x16 = standard_convolution_layer(x15, mask, channel_number, 3, 1, name="conv16")

        x17 = standard_convolution_layer(x16, mask, channel_number // 2, 3, 1, name="conv17")
        x18 = standard_convolution_layer(x17, mask, 3, 3, 1, name="conv18")
        x18 = tf.clip_by_value(x18, -1.0, 1.0)

        return x18


def global_perceiving_network(x, mask, padding="SAME", name="inpaint_net_1", reuse=False):
    """
    Global perceiving network
    Args:
            x: incomplete image
            mask: mask region {0,1}
    returns:
            image predicted by global perceiving network
    """
    channel_number = 32
    x = tf.concat([x, mask], axis=3)
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        x1 = standard_convolution_layer(x, mask, channel_number, 5, 1, name="conv1")
        x2 = standard_convolution_layer(x1, mask, 2 * channel_number, 3, 2, name="conv2_downsample")
        x3 = standard_convolution_layer(x2, mask, 2 * channel_number, 3, 1, name="conv3")
        x4 = standard_convolution_layer(x3, mask, 4 * channel_number, 3, 2, name="conv4_downsample")
        x5 = standard_convolution_layer(x4, mask, 4 * channel_number, 3, 1, name="conv5")
        x6 = standard_convolution_layer(x5, mask, 4 * channel_number, 3, 1, name="conv6")

        # dilated conv
        x7 = standard_convolution_layer(x6, mask, 4 * channel_number, 3, dilation_rate=2, name="conv7_atrous")
        x8 = standard_convolution_layer(x7, mask, 4 * channel_number, 3, dilation_rate=4, name="conv8_atrous")
        x9 = standard_convolution_layer(x8, mask, 4 * channel_number, 3, dilation_rate=8, name="conv9_atrous")
        x10 = standard_convolution_layer(x9, mask, 4 * channel_number, 3, dilation_rate=16, name="conv10_atrous")

        x11 = standard_convolution_layer(
            tf.concat([x10, x6], axis=-1), mask, 4 * channel_number, 3, 1, name="conv11"
        )
        x12 = standard_convolution_layer(
            tf.concat([x11, x5], axis=-1), mask, 4 * channel_number, 3, 1, name="conv12"
        )

        x13 = standard_deconvolution_layer(
            tf.concat([x12, x4], axis=-1), mask, 2 * channel_number, name="conv13_upsample"
        )
        x14 = standard_convolution_layer(
            tf.concat([x13, x3], axis=-1), mask, 2 * channel_number, 3, 1, name="conv14"
        )
        x15 = standard_deconvolution_layer(
            tf.concat([x14, x2], axis=-1), mask, channel_number, name="conv15_upsample"
        )
        x16 = standard_convolution_layer(
            tf.concat([x15, x1], axis=-1), mask, channel_number // 2, 3, 1, name="conv16"
        )
        x17 = standard_convolution_layer(x16, mask, 3, 3, 1, name="conv17")
        x18 = tf.clip_by_value(x17, -1.0, 1.0)

        return x18
