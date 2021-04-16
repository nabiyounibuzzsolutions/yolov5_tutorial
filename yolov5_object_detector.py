
from flask import request, Flask, current_app, make_response
from google.cloud import storage
import torch

#defining local variables
GOOGLE_APPLICATION_CREDENTIALS = [path_to_the_google_applicaiton_credentials]
bucket_name = [name_of_the_bucket_of_the_images_in_the_storage]
model_weights = [path_to_the_trained_weights.pt]
download_path = "/tmp"

# setup the flask apo
app = Flask(__name__) 

# flask enpoint for object detection
@app.route('/detect', methods=["POST"])
def object_detector():
	# reading passed arguments as a json variable
	image_path = request.form.get("image_path")
	current_app.logger.info("\nObject detection for the image {}\n".format(image_path))
	# craeting aclient for uploading images to the GCP storage
	storage_client = storage.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(image_path)
	# downloading the image from the GCP bucket
	current_app.logger.info("\nReading the image from the bucket ...")
	blob.download_to_filename(download_path)
	img = cv2.imread(download_path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # reading the model weights
	current_app.logger.info("\nLoading the object detector ...")
	model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model = model_weights)
	model.conf = 0.01 # confidence threshold
	model.iou = 0.6 # iou threshold
    # passing the image to the model for prediction
	current_app.logger.info("\nDetecting objects of the image...")
	results = model(img).xyxy[0].numpy()
	# returning the detected objects info to the user
	return results

if __name__ == '__main__':
    if not app.config['LOCAL']:
        try:
            import googleclouddebugger
            googleclouddebugger.enable(
                breakpoint_enable_canary=False
            )
        except ImportError:
            pass
    app.debug = True
    with app.app_context():
        app.run(host="localhost", port=8080, debug=True)
