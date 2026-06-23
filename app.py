import importlib

try:
    flask = importlib.import_module("flask")
    Flask = flask.Flask
    render_template = flask.render_template
    request = flask.request
    jsonify = flask.jsonify
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Flask is not installed. Please install it with `pip install flask` "
        "and ensure you are using the correct Python environment."
    ) from e
import cv2
import numpy as np
import base64
import io
import os
from PIL import Image
import dct_compression as dct

app = Flask(__name__)

def array_to_base64(img_array):
    # Ensure it's uint8
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    # Convert grayscale to PIL Image
    pil_img = Image.fromarray(img_array)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def process_image(image, quality):
    # compress
    compressed, q_matrix, padding = dct.compress_image(image, quality)
    # decompress
    reconstructed = dct.decompress_image(compressed, q_matrix, image.shape, padding)
    # metrics
    metrics = dct.compute_metrics(image, reconstructed)
    cr = dct.estimate_compression_ratio(compressed, image)
    
    # difference map
    diff = np.abs(image.astype(np.int16) - reconstructed.astype(np.int16))
    # map difference to a heatmap look (using cv2 colormap)
    diff_uint8 = np.clip(diff * 5, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_HOT)
    # Convert BGR to RGB
    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
    
    # Convert to base64
    orig_b64 = array_to_base64(image)
    recon_b64 = array_to_base64(reconstructed)
    
    # Diff color needs to be saved slightly differently
    pil_diff = Image.fromarray(diff_color)
    buff = io.BytesIO()
    pil_diff.save(buff, format="PNG")
    diff_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    diff_b64 = f"data:image/png;base64,{diff_str}"
    
    return {
        "original": orig_b64,
        "reconstructed": recon_b64,
        "difference": diff_b64,
        "metrics": metrics,
        "compression_ratio": cr
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
        
    quality = int(request.form.get('quality', 50))
    
    # Read image from memory
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400
        
    try:
        result = process_image(image, quality)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
