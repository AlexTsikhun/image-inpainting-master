import tensorflow as tf
from tensorflow.keras.layers import Conv2D


def _pad_tensor_with_reflect_padding(
    input_tensor, kernel_size: int, dilation_rate: int
) -> tf.Tensor:
    """
    Helper function to apply reflective padding to the input tensor.

    Args:
        input_tensor: The input tensor to be padded.
        kernel_size: The kernel size used for convolution.
        dilation_rate: The dilation rate used for convolution.

    Returns:
        Padded tensor.
    """

    padding_size = int(dilation_rate * (kernel_size - 1) / 2)
    return tf.pad(input_tensor, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")


def standard_convolution_layer(
    input_tensor, mask, channel_number, kernel_size=3, stride=1, dilation_rate=1, name="conv", padding="SAME"
):
    """
    define convolution for generator
    Args:
            x:iput image
            cnum: channel number
            ksize: kernal size
            stride: convolution stride
            rate : rate for dilated conv
            name: name of layers
    """
    padded_tensor = _pad_tensor_with_reflect_padding(input_tensor, kernel_size, dilation_rate)
    return Conv2D(
        filters=channel_number,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=dilation_rate,
        activation=tf.nn.elu,
        padding="VALID",
        name=name + "_1",
    )(padded_tensor)


def standard_deconvolution_layer(input_tensor, mask, channel_number, name="deconv", padding="VALID"):
    """
    define upsample convolution for generator
    Args:
    x: input image
    mask: input mask
    name: name of layers
    """

    dilation_rate = 1
    kernel_size = 3
    stride = 1

    input_shape = input_tensor.get_shape().as_list()
    upsampled_tensor = tf.image.resize(input_tensor, [input_shape[1] * 2, input_shape[2] * 2])
    padded_tensor = _pad_tensor_with_reflect_padding(upsampled_tensor, kernel_size=kernel_size, dilation_rate=dilation_rate)

    return Conv2D(
        filters=channel_number,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=dilation_rate,
        activation=tf.nn.elu,
        padding=padding,
        name=name + "_1", #?
    )(padded_tensor)
