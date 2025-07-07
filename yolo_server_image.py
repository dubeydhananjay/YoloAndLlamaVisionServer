import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import cv2
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov9'))

# Add yolov9 directory to the path
from yolov9.models.experimental import attempt_load
from yolov9.utils.general import non_max_suppression

# Load YOLOv9 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model_path = os.path.join(os.path.dirname(__file__), 'yolov9', 'yolov9t.pt')
try:
    if not os.path.exists(yolo_model_path):
        logger.error(f"YOLOv9 model file not found at {yolo_model_path}")
        raise FileNotFoundError(f"Model file missing: {yolo_model_path}")
    yolo_model = attempt_load(yolo_model_path).to(device).eval()
    logger.info(f"YOLOv9 model loaded successfully from {yolo_model_path}")
except Exception as e:
    logger.error(f"Failed to load YOLOv9 model: {e}")
    raise

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

# Directory to save input images
INPUT_IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'input_images')
os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coordinates from img1_shape to img0_shape"""
    logger.debug("Scaling coordinates...")
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
    logger.debug(f"Letterbox: Resizing image to {new_shape}")
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

def multi_scale_detection(image_np, scales=[640, 800, 960], conf_thres=0.05, iou_thres=0.6):
    """Perform detection at multiple scales and aggregate results."""
    logger.debug(f"Performing multi-scale detection with scales: {scales}, conf_thres: {conf_thres}, iou_thres: {iou_thres}")
    results = []
    for scale in scales:
        img_scaled, ratio, (dw, dh) = letterbox(image_np, new_shape=(scale, scale))
        img_tensor = torch.from_numpy(img_scaled[:, :, ::-1].copy().transpose(2, 0, 1)).to(device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        logger.debug(f"Input tensor shape: {img_tensor.shape}")

        with torch.no_grad():
            preds = yolo_model(img_tensor)[0]
        preds = non_max_suppression(preds, conf_thres, iou_thres, agnostic=False)
        logger.debug(f"Predictions for scale {scale}: {len(preds)} detections")

        for det in preds:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], image_np.shape, ratio_pad=(ratio, (dw, dh))).round()
                for *xyxy, conf, cls in det:
                    class_idx = int(cls.item())
                    class_name = class_names[class_idx] if class_idx < len(class_names) else 'unknown'
                    results.append({
                        'class': class_name,
                        'confidence': conf.item(),
                        'x1': xyxy[0].item(),
                        'y1': xyxy[1].item(),
                        'x2': xyxy[2].item(),
                        'y2': xyxy[3].item()
                    })
                    logger.debug(f"Detected: {class_name}, Confidence: {conf.item()}, Box: {xyxy}")
    logger.info(f"Total detections: {len(results)}")
    return results

def process_image(file):
    try:
        logger.info("Processing image...")

        # Read the incoming image data
        image_data = file.read()
        
        # Save the incoming image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        input_image_path = os.path.join(INPUT_IMAGE_DIR, f"input_image_{timestamp}.jpg")
        with open(input_image_path, 'wb') as f:
            f.write(image_data)
        logger.info(f"Input image saved as {input_image_path}")

        # Reset file pointer to start for further processing
        file.seek(0)
        
        # Process the image with YOLOv9
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img_np = np.array(img)
        logger.debug(f"Image loaded, shape: {img_np.shape}")

        detections = multi_scale_detection(img_np, conf_thres=0.05, iou_thres=0.6)
        
        # Prepare the response with cropped images
        response = {'objects': []}
        
        for res in detections:
            # Crop the bounding box from the original image
            x1, y1, x2, y2 = int(res['x1']), int(res['y1']), int(res['x2']), int(res['y2'])
            try:
                cropped_img = img.crop((x1, y1, x2, y2))
                
                # Encode cropped image to base64
                buffered = BytesIO()
                cropped_img.save(buffered, format="JPEG")
                cropped_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Add detection details and cropped image to response
                response['objects'].append({
                    'class': res['class'],
                    'confidence': res['confidence'],
                    'x1': res['x1'],
                    'y1': res['y1'],
                    'x2': res['x2'],
                    'y2': res['y2'],
                    'cropped_image': cropped_img_str
                })
            except Exception as e:
                logger.error(f"Error cropping image: {e}")
                continue
        
        # Draw bounding boxes on the original image
        draw = ImageDraw.Draw(img)
        for res in detections:
            draw.rectangle([res['x1'], res['y1'], res['x2'], res['y2']], outline="red", width=2)
            draw.text((res['x1'], res['y1']), f"{res['class']} {res['confidence']:.2f}", fill="red")
        
        # Encode the annotated original image to base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        response['image'] = img_str
        logger.info(f"Response prepared with {len(response['objects'])} objects")
        return response
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return {'objects': [], 'error': str(e)}