import uuid
import os
import cv2
import base64
from flask import render_template, jsonify, request
from flask import Flask
from PIL import Image
from download import download_all_weights
from inpaint.predict import InpaintingModel
from config import config

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])

application = Flask(__name__)
application.config["DEBUG"] = True
application.config["CACHE_TYPE"] = "null"
application.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

download_all_weights()

inpainting_model = InpaintingModel()


@application.route("/")
def home_page():
    return render_template("index.html")


def resize_image(path, size=(512, 512)):
    image = Image.open(path)
    resized_image = image.resize(size, Image.LANCZOS)
    resized_image.save(path)


@application.route("/process-image", methods=["POST"])
def process_image():
    try:
        filename = str(uuid.uuid4())
        file_path_raw = os.path.join(application.config["UPLOAD_FOLDER"], filename + ".png")
        file_path_mask = os.path.join(
            application.config["UPLOAD_FOLDER"], "mask_" + filename + ".png"
        )
        file_path_output = os.path.join(
            application.config["UPLOAD_FOLDER"], "output_" + filename + ".png"
        )

        mask_base64_string = request.values[("mask_b64")]
        mask_image_data = mask_base64_string.split(",")[1]

        with open(file_path_mask, "wb") as mask_file:
            mask_file.write(base64.b64decode(mask_image_data))

        resize_image(file_path_mask)

        file_raw = request.files.get("input_file")
        file_raw.save(file_path_raw)
        resize_image(file_path_raw)

        # Region Wise
        inpainted_image = inpainting_model.run_fill(file_path_raw, file_path_mask)

        cv2.imwrite(file_path_output, inpainted_image)
        resize_image(file_path_output, size=(384, 384))

        return jsonify(
            {
                "output_image": os.path.join(
                    "static", "uploads", os.path.basename(file_path_output)
                ),
            }
        )
    except Exception as error:
        print("Errrrrrrrrrrrr", error)
        return jsonify({"status": "error"})


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=config.PORT, use_reloader=True, threaded=False)
