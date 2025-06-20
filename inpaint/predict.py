import cv2
import numpy as np
import tensorflow as tf

from .inpaint_model import region_wise_generator

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MODEL_PATH = "inpaint/models/places2.ckpt-6666"


class InpaintingModel:
    def __init__(self):
        """Initialize TensorFlow session and model."""
        tf.compat.v1.disable_eager_execution()
        self.tensorflow_session = tf.compat.v1.Session()
        self._build_model()

    def _build_model(self):
        """Set up placeholders and model graph."""
        self.images = tf.compat.v1.placeholder(
            tf.float32, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="image"
        )
        self.mask = tf.compat.v1.placeholder(
            tf.float32, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="mask"
        )
        self.inpainting_result = self._inference(self.images, self.mask)

        init_operation = tf.group(
            tf.compat.v1.initialize_all_variables(),
            tf.compat.v1.initialize_local_variables(),
        )
        self.tensorflow_session.run(init_operation)

        saver = tf.compat.v1.train.Saver()
        saver.restore(self.tensorflow_session, MODEL_PATH)

    def _inference(self, batch_data, mask, reuse=False):
        """Inference logic for inpainting."""
        batch_normalized = batch_data / 127.5 - 1.0
        batch_incomplete = batch_normalized * mask
        predicted_image = region_wise_generator(batch_incomplete, mask, reuse=reuse)
        inpainted_image = batch_incomplete * mask + predicted_image * (1.0 - mask)
        return (inpainted_image + 1.0) * 127.5

    def run_fill(self, image_path, mask_path):
        """Process image and mask for inpainting."""
        try:
            mask = cv2.resize(cv2.imread(mask_path), (IMAGE_HEIGHT, IMAGE_WIDTH))[:, :, 0:1]
            mask = (mask // 255).astype(np.float32)
            mask = np.where(mask >= 0.5, 1.0, 0.0)
            mask = 1.0 - mask
            mask = np.expand_dims(mask, 0)

            input_image = cv2.imread(image_path)[..., ::-1]
            input_image = cv2.resize(input_image, (IMAGE_HEIGHT, IMAGE_WIDTH))
            input_image = np.expand_dims(input_image, 0)

            # Run inference
            inpainting_result = self.tensorflow_session.run(
                self.inpainting_result, feed_dict={self.mask: mask, self.images: input_image}
            )
            return inpainting_result[0][..., ::-1]

        except Exception as e:
            raise RuntimeError(f"Error during inpainting: {str(e)}")

    def __del__(self):
        """Clean up session."""
        self.tensorflow_session.close()
