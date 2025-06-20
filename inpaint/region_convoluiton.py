import tensorflow as tf
from .inpaint_convolution import standard_deconvolution_layer
from tensorflow.keras.layers import Conv2D

# tensor = x
def region_deconvolution(complete_tensor, missing_tensor, mask_tensor, layer_name="com_"):
    """Perform region-wise deconvolution to fuse complete and missing image regions.
    
    Args:
        complete_tensor: Tensor representing the complete image regions.
        missing_tensor: Tensor representing the missing image regions.
        mask_tensor: Binary mask tensor (1 for known, 0 for missing).
        layer_name: Prefix for layer naming (default: 'composite').

    Returns:
        Fused tensor after deconvolution.
    """

    complete_input_shape = complete_tensor.get_shape().as_list()

    resized_mask_tensor = tf.image.resize(mask_tensor, size=[complete_input_shape[1], complete_input_shape[2]])
    reshaped_mask_tensor = tf.reshape(resized_mask_tensor, [complete_input_shape[0], complete_input_shape[1], complete_input_shape[2], 1])

    masked_complete_tensor = complete_tensor * reshaped_mask_tensor

    missing_input_shape = missing_tensor.get_shape().as_list()
    masked_missing_tensor = missing_tensor * (1.0 - reshaped_mask_tensor)
    
    fused_tensor = tf.concat([masked_complete_tensor, masked_missing_tensor], axis=-1)

    return standard_deconvolution_layer(fused_tensor, reshaped_mask_tensor, missing_input_shape[-1], name=layer_name + "_fusion")


def region_convolution(complete_tensor, missing_tensor, mask_tensor, layer_name="com_"):
    """Perform region-wise convolution to fuse complete and missing image regions.
    
    Args:
        complete_tensor: Tensor representing the complete image regions.
        missing_tensor: Tensor representing the missing image regions.
        mask_tensor: Binary mask tensor (1 for known, 0 for missing).
        layer_name: Prefix for layer naming (default: 'composite').

    Returns:
        Fused tensor after convolution and ELU activation.
    """

    complete_tensor_shape = complete_tensor.get_shape().as_list()
    resized_mask_tensor = tf.image.resize(mask_tensor, size=[complete_tensor_shape[1], complete_tensor_shape[2]])
    reshaped_mask_tensor = tf.reshape(resized_mask_tensor, [complete_tensor_shape[0], complete_tensor_shape[1], complete_tensor_shape[2], 1])

    padding_size = int(1 * (3 - 1) / 2)

    masked_complete_tensor = complete_tensor * reshaped_mask_tensor
    missing_tensor_shape = missing_tensor.get_shape().as_list()
    masked_missing_tensor = missing_tensor * (1.0 - reshaped_mask_tensor)
    fused_tensor = tf.concat([masked_complete_tensor, masked_missing_tensor], axis=-1)
    fused_tensor = tf.pad(fused_tensor, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], "REFLECT")

    fused_tensor = Conv2D(
        filters=missing_tensor_shape[-1],
        kernel_size=3,
        strides=1,
        padding="valid",
        dilation_rate=1,
        activation=None,
        name=layer_name + "_fusion",
    )(fused_tensor)
    return tf.nn.elu(fused_tensor)
