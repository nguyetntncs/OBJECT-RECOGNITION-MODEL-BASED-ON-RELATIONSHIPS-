import torch
import json
import clip
import cv2
import sys
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# Load mô hình YOLOv5 (pretrained)
yolo_model = YOLO(r"D:\book\PTIT\test\yolov5\yolov5su.pt")  # Hoặc yolov5m.pt, yolov5l.pt nếu muốn model mạnh hơn

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Danh sách từ vựng mở rộng (có thể tùy chỉnh)
animals = [
    "cat", "dog", "bird", "horse", "cow", " sheep", "lion",
    "tiger", "elephant", "bear", "deer", "monkey", "zebra",
    "giraffe", "kangaroo", "dolphin", "shark", "snake", "turtle",
    "rabbit", "fox", "wolf", "panda", "crocodile", "peacock","person"
]
vehicles = [
    "car", "motorcycle", "bicycle", " bus", " truck", " train",
    "airplane", " helicopter", "boat", " yacht", " submarine",
    " scooter", " skateboard", "tram", " taxi", " police car",
    " ambulance", " fire truck", " forklift", " van"
]
household_items = [
    " chair", " table", " sofa", " bed", " lamp", " television",
    " laptop", " smartphone", " refrigerator", " microwave", " washing machine",
    "vacuum cleaner", " mirror", " bookshelf", " fan", " clock",
    " pillow", " blanket", " rug", " cupboard", " kettle", " toaster","key"
]
food_drinks = [
    " pizza", " hamburger", " sandwich", " hotdog", " steak", " fish",
    "bowl of soup", " plate of spaghetti", " salad", " cake", " donut",
    " ice cream", " cup of coffee", " bottle of water", " soda can",
    " glass of milk", " loaf of bread", " croissant", " chocolate bar"
]
clothing = [
    "t-shirt", " shirt", " pair of jeans", " dress", "skirt", " jacket",
    " coat", " pair of shorts", " hat", " cap", " pair of sunglasses",
    "pair of gloves", " belt", " scarf", " backpack", " handbag",
    " pair of shoes", " pair of sandals", " pair of boots"
]
electronics = [
    " smartphone", " laptop", " desktop computer", " keyboard", " mouse",
    " printer", " projector", " television", " camera", " drone",
    " game console", " tablet", " smartwatch", " microphone",
    " speaker", " headphone", " charger", " USB drive", " hard drive"
]
buildings = [
    "house", " skyscraper", " bridge", " tower", " lighthouse",
    " castle", " temple", "church", " mosque", " stadium",
    " factory", " warehouse", " hospital", "school", " shopping mall",
    " hotel", " police station", " fire station", " library", "a museum"
]
nature = [
    "mountain", " river", " lake", " ocean", " beach", " desert",
    "forest", " waterfall", " volcano", "cave", " rainbow",
    " sunset", " sunrise", " thunderstorm", "snow-covered mountain",
    " flower", " tree", " bush", " meadow", " glacier"
]
sports = [
    "soccer ball", " basketball", " baseball bat", " tennis racket",
    " golf club", " hockey stick", " snowboard", " skateboard",
    " pair of ice skates", " bicycle helmet", " football", " volleyball",
    " badminton racket", " boxing glove", " jump rope"
]
school_supplies = [
    " book", " notebook", " pen", " pencil", " eraser", " ruler",
    " calculator", " protractor", " compass", " highlighter", 
    " stapler", " pair of scissors", " glue stick", " backpack",
    " whiteboard", " blackboard", " piece of chalk", " marker",
    " set of colored pencils", "paintbrush", " watercolor palette",
    " binder", " paper clip", " sticky note", " file folder",
    " document scanner", " desk lamp", "tablet", " laptop", 
    " printer", " USB flash drive"
]

# Loại bỏ dấu cách thừa và tránh trùng nhãn
label_texts = list(set([label.strip() for label in (
    animals + vehicles + household_items +
    food_drinks + clothing + electronics +
    buildings + nature + sports + school_supplies
)]))

text_inputs = clip.tokenize(label_texts).to(device)

def add_padding(image, bbox, padding=10):
    x1, y1, x2, y2 = bbox
    img_height, img_width = image.shape[:2]
    return image[max(0, y1-padding):min(y2+padding, img_height), max(0, x1-padding):min(x2+padding, img_width)]

# Detect objects with YOLO
def detect_objects(image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image_path)
    detected_objects, yolo_labels = [], []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            if (x2 - x1 < 20) or (y2 - y1 < 20):
                print(f"⚠️ Bỏ qua đối tượng nhỏ quá [{x1}, {y1}, {x2}, {y2}]")
                continue
            cropped_pil = Image.fromarray(cv2.cvtColor(add_padding(image, (x1, y1, x2, y2)), cv2.COLOR_BGR2RGB))
            detected_objects.append((cropped_pil, (x1, y1, x2, y2)))
            yolo_labels.append(results[0].names[int(cls)])
    return detected_objects, yolo_labels, image

# Phân loại với CLIP, fallback về YOLO nếu confidence thấp
def classify_with_clip(detected_objects, yolo_labels):
    results = []
    text_features = clip_model.encode_text(text_inputs)

    # 🌟 Lấy toàn bộ nhãn từ YOLO làm nhãn quan trọng
    important_labels = list(yolo_model.names.values())
    print(f"🔹 Danh sách nhãn quan trọng từ YOLO: {important_labels}")

    for idx, (cropped_pil, bbox) in enumerate(detected_objects):
        try:
            if not isinstance(cropped_pil, Image.Image):
                print(f"⚠️ Đối tượng {idx+1} không phải ảnh PIL! Giữ nhãn YOLO: '{yolo_labels[idx]}'")
                results.append((yolo_labels[idx], bbox))
                continue

            # Đưa ảnh vào CLIP để phân loại
            image_input = preprocess(cropped_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                similarities = (clip_model.encode_image(image_input) @ text_features.T).softmax(dim=-1)
                best_label = label_texts[similarities.argmax().item()]
                confidence = similarities.max().item()

            # 🎯 Logic thông minh giữ nhãn YOLO nếu CLIP nhận sai
            if yolo_labels[idx] in important_labels and best_label != yolo_labels[idx]:
                print(f"🔹 Giữ nhãn YOLO (quan trọng): '{yolo_labels[idx]}' dù CLIP báo '{best_label}' (conf: {confidence:.2f})")
                results.append((yolo_labels[idx], bbox))
            elif confidence < 0.3:
                print(f"⚠️ Độ tự tin thấp ({confidence:.2f}) → Giữ nhãn YOLO: '{yolo_labels[idx]}'")
                results.append((yolo_labels[idx], bbox))
            else:
                print(f"✅ Đổi nhãn CLIP: '{yolo_labels[idx]}' ➜ '{best_label}' (confidence: {confidence:.2f})")
                results.append((best_label.strip(), bbox))

        except Exception as e:
            print(f"⚠️ Lỗi CLIP: {e} → Giữ nhãn YOLO: '{yolo_labels[idx]}'")
            results.append((yolo_labels[idx], bbox))

    return results

# Full pipeline
def run_pipeline(image_path=None):
    if image_path is None:
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(title="Chọn file ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not image_path:
            print("❌ Không có file nào được chọn!")
            return

    detected_objects, yolo_labels, original_image = detect_objects(image_path)
    classified_results = classify_with_clip(detected_objects, yolo_labels)

    results_json = []
    for idx, (label, (x1, y1, x2, y2)) in enumerate(classified_results):
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        results_json.append({"label": label, "bbox": [int(x1), int(y1), int(x2), int(y2)]})
        print(f"✅ Đối tượng {idx+1} | Class: {label} | BBox: [{x1}, {y1}, {x2}, {y2}]")

    with open("result.json", "w") as json_file:
        json.dump(results_json, json_file, indent=4)

    cv2.imwrite("result.jpg", original_image)
    print("✅ Nhận diện hoàn tất! Kết quả đã lưu.")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(image_path)
