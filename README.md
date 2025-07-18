# 🦙🔍 Yolo + LLaMA Vision Server

This project integrates **YOLOv9 object detection** and **Meta’s LLaMA Vision-Instruct language model** into a unified REST API service.

With this system, you can:
- Detect objects in images with YOLOv9.
- Generate contextual descriptions using LLaMA.
- Combine vision and language to build intelligent applications (e.g., Mixed Reality, assistive systems).

---

## ✨ Features

✅ **YOLOv9 Detection**
- Multi-scale detection for improved accuracy.
- Bounding box extraction and labeling.
- Cropped image thumbnails (base64).

✅ **LLaMA Vision-Instruct Descriptions**
- Generate natural language descriptions of images or detected regions.

✅ **REST API Server**
- Endpoints to process images and generate text.

✅ **Designed for Integration**
- Easily connect with Unity or web clients.

---

## 🛠️ Installation

> **Note:** You need a machine with **CUDA-compatible GPU** and **Python 3.8+**.

---

### 1️⃣ Clone the repository

```bash
git clone https://github.com/dubeydhananjay/YoloAndLlamaVision.git
cd YoloAndLlamaVision
```

---

### 2️⃣ Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

---

### 3️⃣ Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download YOLOv9 model weights

Download `yolov9t.pt` (or your preferred YOLOv9 variant) from the [official YOLOv9 repository](https://github.com/WongKinYiu/yolov9).  
Place the `.pt` file in:

```
yolov9/yolov9t.pt
```

✅ Example:

```
YoloAndLlamaVision/
├── yolov9/
│   └── yolov9t.pt
```

---

### 5️⃣ Configure LLaMA

This project uses:

```
meta-llama/Llama-3.2-11B-Vision-Instruct
```

Make sure you have access to this model in Hugging Face.  
If needed, login:

```bash
huggingface-cli login
```

---

## 🚀 Running the Server

```bash
python main_server.py
```

✅ By default, the server runs at:

```
http://localhost:5001
```

---

## 🎯 API Reference

---

### 🟢 **POST /process**

Detects objects in an image.

**Query Parameter:**
```
type=image
```

**Form Data:**
- `file`: Image file (JPEG/PNG)

✅ **Example request:**

```bash
curl -X POST "http://localhost:5001/process?type=image" \
     -F "file=@your_image.jpg"
```

**Response JSON:**

```json
{
  "objects": [
    {
      "class": "dog",
      "confidence": 0.87,
      "x1": 100,
      "y1": 150,
      "x2": 300,
      "y2": 400,
      "cropped_image": "base64-encoded-jpeg"
    }
  ],
  "image": "base64-encoded-jpeg-with-boxes"
}
```

---

### 🟢 **POST /generate_description**

Generates a text description of an image.

**Form Data:**
- `image`: Image file
- `prompt`: Text prompt to guide description

✅ **Example request:**

```bash
curl -X POST "http://localhost:5001/generate_description" \
     -F "image=@your_image.jpg" \
     -F "prompt=Describe the objects in this image."
```

**Response JSON:**

```json
{
  "description": "The image shows a dog sitting on a couch."
}
```

---

## 📂 Project Structure

```
YoloAndLlamaVision/
│
├── main_server.py          # Flask API server
├── yolo_server_image.py    # YOLOv9 image processing logic
├── llama_server.py         # LLaMA description generation
├── yolov9/                 # YOLOv9 code and weights
│   ├── models/
│   ├── utils/
│   └── yolov9t.pt
├── input_images/           # Saved uploaded images
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- CUDA-compatible GPU (for YOLOv9 and LLaMA acceleration)
- [YOLOv9 weights](https://github.com/WongKinYiu/yolov9)
- Hugging Face access to LLaMA Vision-Instruct

---

## 🧠 Acknowledgements

- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- Meta AI LLaMA
- Hugging Face Transformers
- Ultralytics YOLO community

---

## 📄 License

MIT License

---

## 🙌 Contributing

Pull requests and issues are welcome!  
If you’d like to contribute, please fork the repository and open a PR.

---

## ✉️ Contact

Feel free to reach out via [GitHub Issues](https://github.com/dubeydhananjay/YoloAndLlamaVision/issues).

---
