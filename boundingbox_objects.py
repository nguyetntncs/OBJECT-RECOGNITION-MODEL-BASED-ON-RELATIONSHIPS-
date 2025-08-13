import json
import os
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from models import build_model

def load_yolo_output(json_path):
    """Load object detection results from YOLOv5 JSON output."""
    with open(json_path, 'r') as f:
        yolo_data = json.load(f)

    all_objects = []  # Danh sách chứa tất cả các đối tượng sau khi xử lý

    for image_entry in yolo_data:
        image_id = image_entry.get("image_id", "unknown")  # Lấy tên ảnh, nếu thiếu thì gán 'unknown'
        objects_list = image_entry.get("objects", [])  # Lấy danh sách objects, nếu không có thì gán []

        print(f"Processing image: {image_id}")  # Debug

        # Nếu không có objects, tiếp tục vòng lặp
        if not objects_list:
            print(f"LỖI: Không có objects trong ảnh {image_id}")
            continue

        all_objects.append((image_id, objects_list))  # Lưu danh sách objects cùng với image_id

    return all_objects

def convert_yolo_to_reltr(objects_list, img_size):
    """Convert YOLO bounding boxes to RelTR format (normalized cx, cy, w, h)."""
    img_w, img_h = img_size
    objects = []

    for obj in objects_list:
        print("DEBUG:", obj)  # Chỉ in từng object để tránh lỗi in cả danh sách lớn

        if "bbox" not in obj or "class" not in obj:
            continue
        
        x_min, y_min, x_max, y_max = obj["bbox"]
        x_c = (x_min + x_max) / 2 / img_w
        y_c = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        objects.append({
            "label": obj["class"],
            "bbox": [x_c, y_c, w, h],
            "group_id": hash(obj["class"])  # Nhóm các vật thể cùng loại
        })
    return objects

def run_reltr_inference(objects, img_path, args, output_json="relationships.json"):
    """Run RelTR to infer relationships between detected objects and save results to a JSON file."""
    if len(objects) < 2:
        print("Lưu ý: Không đủ vật thể để dự đoán quan hệ!")
        return []

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                   'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                   'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                   'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                   'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                   'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    model, _, _ = build_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    rel_logits = outputs["rel_logits"].softmax(-1)[0, :, :-1]
    keep = rel_logits.max(-1).values > 0.1
    rel_indices = rel_logits[keep].argmax(-1)

    relationships = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            subj = objects[i]["label"]
            obj = objects[j]["label"]
            relation = REL_CLASSES[rel_indices[i % len(rel_indices)].item()]
            
            relationships.append({
                "subject": subj,
                "relation": relation,
                "object": obj
            })

    # Ghi kết quả ra file JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(relationships, f, indent=4, ensure_ascii=False)

    print(f"Kết quả đã được lưu vào {output_json}")

    return relationships

def draw_relationships(image_path, objects_list, relationships, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể mở hình ảnh: {image_path}")
        return

    # Vẽ bounding box cho các đối tượng
    for obj in objects_list:
        bbox = obj["bbox"]
        label = obj.get("class", "Unknown")
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Vẽ các mối quan hệ giữa các đối tượng
    for rel in relationships:
        subject_name = rel["subject"]
        object_name = rel["object"]
        relation = rel["relation"]

        subject_obj = next((obj for obj in objects_list if obj["class"] == subject_name), None)
        object_obj = next((obj for obj in objects_list if obj["class"] == object_name), None)

        if subject_obj and object_obj:
            x1, y1, _, _ = subject_obj["bbox"]
            x2, y2, _, _ = object_obj["bbox"]
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(image, relation, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            print(f"Lỗi: Không tìm thấy subject ({subject_name}) hoặc object ({object_name})")

    # 🌟 Đảm bảo ảnh được lưu cùng thư mục ảnh gốc
    if output_path is None:
        output_dir = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"output_{image_name}.jpg")

    cv2.imwrite(output_path, image)
    print(f"✅ Ảnh đã lưu tại {output_path}")

    image = cv2.imread(image_path)

def main():
    parser = argparse.ArgumentParser('YOLO-RelTR Pipeline')
    parser.add_argument('--yolo_json', type=str, default='bboxes_for_reltr.json', help='Path to YOLOv5 JSON output')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg', type=str, help='Dataset type (vg or oi)')
    parser.add_argument('--img_path', type=str, required=True, help='Path to input image')  # 🔥 Thêm đường dẫn ảnh từ `app.py`


    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")

    args = parser.parse_args()

     # Kiểm tra file JSON đầu vào
    if not os.path.exists(args.yolo_json):
        print(f"❌ Không tìm thấy file JSON: {args.yolo_json}")
        return

    print(f"📂 Sử dụng file JSON: {args.yolo_json}")

    # Kiểm tra checkpoint có tồn tại không
    if not os.path.exists(args.resume):
        print(f"❌ Không tìm thấy checkpoint: {args.resume}")
        return

    print(f"✅ Sử dụng checkpoint: {args.resume}")

      # Kiểm tra ảnh đầu vào
    if not os.path.exists(args.img_path):
        print(f"❌ Không tìm thấy ảnh: {args.img_path}")
        return

    # Đọc file JSON
    with open(args.yolo_json, "r") as file:
        yolo_data = json.load(file)

    print(f"🔍 Đọc thành công {len(yolo_data[0]['objects'])} đối tượng từ {args.yolo_json}")

    
    yolo_data = load_yolo_output(args.yolo_json)
    
    all_relationships = {}  # Dictionary để lưu kết quả của từng ảnh

    for image_id, objects_list in yolo_data:
        img = Image.open(args.img_path)  # 🔥 Load ảnh từ `app.py`
        objects = convert_yolo_to_reltr(objects_list, img.size)  # Chuyển đổi format

        if len(objects) < 2:
            print(f"Lưu ý: Ảnh {image_id} có {len(objects)} vật thể, bỏ qua!")  
            continue  # Bỏ qua ảnh này nếu không có đủ vật thể

        relationships = run_reltr_inference(objects, args.img_path, args)  # Chạy RelTR
        all_relationships[image_id] = relationships

        image_id = os.path.splitext(os.path.basename(args.img_path))[0]  # Xóa phần mở rộng .jpg
        output_path = f"output_{image_id}.jpg"
        draw_relationships(args.img_path, objects_list, relationships, output_path)

    
    print(json.dumps(all_relationships, indent=2))

if __name__ == '__main__':
    main()
