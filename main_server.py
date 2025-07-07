# main_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from yolo_server_image import process_image
from llama_server import generate_image_description
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process_file():
    if 'type' not in request.args:
        return jsonify({'error': 'No type specified'}), 400

    file_type = request.args['type']
    file = request.files['file']

    if file_type == 'image':
        return jsonify(process_image(file))
    else:
        return jsonify({'error': 'Invalid type specified'}), 400

@app.route('/generate_description', methods=['POST'])
def generate_description():
    if 'image' not in request.files or 'prompt' not in request.form:
        print("error: Image or prompt not provided.")
        return jsonify({'error': 'Image or prompt not provided.'}), 400

    prompt = request.form['prompt']
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')

    description = generate_image_description(image, prompt)
    return jsonify({'description': description})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
