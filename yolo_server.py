# yolo_server.py
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov9'))

# Add yolov9 directory to the path
from yolov9.models.experimental import attempt_load
from yolov9.utils.general import non_max_suppression

# Load YOLOv9 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model_path = os.path.join(os.path.dirname(__file__), 'yolov9', 'yolov9t.pt')
yolo_model = attempt_load(yolo_model_path).to(device).eval()

# COCO class labels
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coordinates from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img1_shape and img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resizing and padding for YOLO input
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, int(round(dh - 0.1)), int(round(dh + 0.1)), int(round(dw - 0.1)), int(round(dw + 0.1)), cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def multi_scale_detection(image_np, scales=[640, 800, 960]):
    """Perform detection at multiple scales and aggregate results."""
    results = []
    for scale in scales:
        img_scaled, ratio, (dw, dh) = letterbox(image_np, new_shape=(scale, scale))
        img_tensor = torch.from_numpy(img_scaled[:, :, ::-1].copy().transpose(2, 0, 1)).to(device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            preds = yolo_model(img_tensor)[0]
        preds = non_max_suppression(preds, 0.25, 0.45, agnostic=False)

        for det in preds:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], image_np.shape, ratio_pad=(ratio, (dw, dh))).round()
                for *xyxy, conf, cls in det:
                    results.append({
                        'class': class_names[int(cls.item())] if int(cls.item()) < len(class_names) else 'unknown',
                        'confidence': conf.item(),
                        'x1': xyxy[0].item(),
                        'y1': xyxy[1].item(),
                        'x2': xyxy[2].item(),
                        'y2': xyxy[3].item()
                    })
    return results

def process_image(file):
    img = Image.open(BytesIO(file.read())).convert('RGB')
    img_np = np.array(img)
    detections = multi_scale_detection(img_np)
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(img)
    for res in detections:
        draw.rectangle([res['x1'], res['y1'], res['x2'], res['y2']], outline="red", width=2)
        draw.text((res['x1'], res['y1']), f"{res['class']} {res['confidence']:.2f}", fill="red")

    # Encode result image to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {'objects': detections, 'image': img_str}
