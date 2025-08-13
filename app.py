import json
import subprocess
import threading
import re
import tkinter as tk
from tkinter import Entry, filedialog, Label, Button, Canvas
from PIL import Image, ImageTk, ImageDraw
import os
import glob
from sentence_transformers import SentenceTransformer, util

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection & Relationship")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        # Tiêu đề
        self.label = tk.Label(root, text="Chọn ảnh để phát hiện vật thể", font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#333")
        self.label.pack(pady=10)

        # Frame chứa các nút chọn ảnh và checkpoint
        frame_top = tk.Frame(root, bg="#f0f0f0")
        frame_top.pack(pady=10)

        self.btn_select = tk.Button(frame_top, text="📁 Chọn ảnh", command=self.select_image, font=("Arial", 12), bg="#007bff", fg="white", width=18)
        self.btn_select.grid(row=0, column=0, padx=10)

        self.btn_checkpoint = tk.Button(frame_top, text="📂 Chọn checkpoint", command=self.select_checkpoint, font=("Arial", 12), bg="#6c757d", fg="white", width=18)
        self.btn_checkpoint.grid(row=0, column=1, padx=10)

        self.btn_run = tk.Button(frame_top, text="▶️ Chạy nhận diện", command=self.run_pipeline_thread, font=("Arial", 12), bg="#28a745", fg="white", width=18)
        self.btn_run.grid(row=0, column=2, padx=10)

        # Ô nhập truy vấn và nút tìm kiếm
        frame_search = tk.Frame(root, bg="#f0f0f0")
        frame_search.pack(pady=10)

        self.entry_query = tk.Entry(frame_search, font=("Arial", 12), width=40, fg="#555")
        self.entry_query.insert(0, "Nhập truy vấn tìm kiếm...")

        # Gắn sự kiện khi click vào ô nhập
        self.entry_query.bind("<FocusIn>", self.clear_placeholder)
        self.entry_query.bind("<FocusOut>", self.restore_placeholder)
        self.entry_query.grid(row=0, column=0, padx=5)

        self.btn_search = tk.Button(frame_search, text="🔍 Tìm vật thể", command=self.search_object, font=("Arial", 12), bg="#ffc107", fg="black", width=18)
        self.btn_search.grid(row=0, column=1, padx=5)

        # Khung hiển thị ảnh
        self.canvas = tk.Canvas(root, width=700, height=450, bg="white", relief="solid", bd=1)
        self.canvas.pack(pady=10)

        # Các đường dẫn mặc định
        self.image_path = None
        self.result_image_path = "result.jpg"
        self.result_json_path = "converted_bboxes.json"
        self.relationship_json_path = "relationships.json"
        self.checkpoint_path = "ckpt/checkpoint.pth"

        # Load model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def clear_placeholder(self, event):
        """ Xóa chữ mặc định khi click vào ô nhập """
        if self.entry_query.get() == "Nhập truy vấn tìm kiếm...":
            self.entry_query.delete(0, tk.END)
            self.entry_query.config(fg="black")  # Chuyển màu chữ về bình thường

    def restore_placeholder(self, event):
        """ Nếu ô trống khi mất focus, hiển thị lại chữ mặc định """
        if not self.entry_query.get():
            self.entry_query.insert(0, "Nhập truy vấn tìm kiếm...")
            self.entry_query.config(fg="#555")  # Màu chữ xám nhạt

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)

    def display_image(self, path):
        image = Image.open(path)
        image = image.resize((600, 400), Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(300, 200, image=self.img_tk)

    def select_checkpoint(self):
        file_path = filedialog.askopenfilename(title="Chọn checkpoint", filetypes=[("Checkpoint files", "*.pth")])
        if file_path:
            self.checkpoint_path = file_path
            print(f"📝 Checkpoint được chọn: {self.checkpoint_path}")

    def search_object(self):
        if not self.image_path:
            self.label.config(text="❌ Hãy chọn ảnh trước!")
            return

        query = self.entry_query.get().strip()
        if not query or query == "Nhập truy vấn tìm kiếm...":
            self.label.config(text="❌ Hãy nhập câu truy vấn!")
            return

        try:
            with open(self.result_json_path, "r") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                data = data[0]

            objects = data.get("objects", [])
            if not objects:
                self.label.config(text="❌ Không có object nào trong JSON!")
                return

            object_names = [obj.get("class", "") for obj in objects]
            query_lower = query.lower()

            # Tìm object có trong câu truy vấn (ưu tiên vị trí sớm nhất)
            best_match_name = None
            best_pos = len(query_lower) + 1  # Giá trị rất lớn

            for obj_name in object_names:
                pos = query_lower.find(obj_name.lower())
                if 0 <= pos < best_pos:
                    best_pos = pos
                    best_match_name = obj_name

            if not best_match_name:
                self.label.config(text="❌ Không tìm thấy vật thể nào trùng trong truy vấn!")
                return

            self.label.config(text=f"✅ Đã tìm thấy vật thể: {best_match_name}")
            print(f"🔍 Vật thể khớp với truy vấn: {best_match_name}")
            self.draw_relationship_boxes(best_match_name, None)

        except json.JSONDecodeError:
            self.label.config(text="❌ Lỗi: Không đọc được file JSON!")
        except Exception as e:
            self.label.config(text=f"❌ Lỗi chi tiết: {e}")


    def draw_relationship_boxes(self, subject_name, object_name):
        try:
            with open(self.result_json_path, "r") as f:
                data = json.load(f)

            print("✅ JSON data loaded:", data)

            if isinstance(data, list):
                data = data[0]

            objects = data.get("objects", [])

            print("🔍 Danh sách objects:", [obj.get("class", "") for obj in objects])
            print("🔍 Subject cần tìm:", subject_name, "| Object cần tìm:", object_name)

            # Tìm subject
            subject_box = next(
                (obj for obj in objects if obj.get("class", "").lower() == subject_name.lower()), None
            )

            # Tìm object nếu có
            object_box = None
            if object_name:
                object_box = next(
                    (obj for obj in objects if obj.get("class", "").lower() == object_name.lower()), None
                )

            if not subject_box:
                self.label.config(text=f"❌ Không tìm thấy subject: {subject_name} trong JSON!")
                print("❌ Lỗi tìm subject:", subject_name)
                return

            if object_name and not object_box:
                self.label.config(text=f"❌ Không tìm thấy object: {object_name} trong JSON!")
                print("❌ Lỗi tìm object:", object_name)
                return

            image = Image.open(self.image_path)
            draw = ImageDraw.Draw(image)

            # Vẽ subject
            sx, sy, sw, sh = subject_box["bbox"]
            draw.rectangle([sx, sy, sw, sh], outline="red", width=3)
            draw.text((sx, sy - 10), subject_name, fill="red")

            # Vẽ object nếu có
            if object_box:
                ox, oy, ow, oh = object_box["bbox"]
                draw.rectangle([ox, oy, ow, oh], outline="blue", width=3)
                draw.text((ox, oy - 10), object_name, fill="blue")

            result_path = "relationship_result.jpg"
            image.save(result_path)
            self.display_image(result_path)

            self.label.config(text="✅ Đã vẽ xong box!")

        except Exception as e:
            self.label.config(text=f"❌ Lỗi khi vẽ box: {e}")
            print(f"❌ Lỗi khi vẽ box: {e}")

    def run_pipeline_thread(self):
        thread = threading.Thread(target=self.run_pipeline)
        thread.start()

    def run_pipeline(self):
        if not self.image_path:
            self.label.config(text="❌ Hãy chọn ảnh trước!")
            return

        self.label.config(text="⏳ Đang xử lý... Vui lòng chờ.")

        try:
            # 1️⃣ Chạy detect_objects.py
            self.label.config(text="🔍 Đang phát hiện vật thể...")
            detect_thread = threading.Thread(target=subprocess.run, args=(["python", "detect_objects.py", self.image_path],))
            detect_thread.start()
            detect_thread.join()  # Đợi detect_objects.py chạy xong

            # 2️⃣ Chạy convert_yolo_to_reltr.py (sau khi detect_objects.py hoàn tất)
            self.label.config(text="🔄 Đang chuyển đổi dữ liệu YOLO...")
            convert_thread = threading.Thread(target=subprocess.run, args=(["python", "convert_yolo_to_reltr.py", "result.json"],))
            convert_thread.start()
            convert_thread.join()  # Đợi convert_yolo_to_reltr.py chạy xong

            # 3️⃣ Chạy boundingbox_objects.py (sau khi convert_yolo_to_reltr.py hoàn tất)
            self.label.config(text="🔗 Đang xác định mối quan hệ giữa các vật thể...")
            boundingbox_thread = threading.Thread(target=subprocess.run, args=(["python", "boundingbox_objects.py", "--yolo_json", self.result_json_path,"--img_path",self.image_path,"--device","cpu", "--resume", self.checkpoint_path],))
            boundingbox_thread.start()
            boundingbox_thread.join()  # Đợi boundingbox_objects.py chạy xong

            image_dir = os.path.dirname(self.image_path)
            image_id = os.path.splitext(os.path.basename(self.image_path))[0]

            # ✅ Tìm ảnh output_anh2.jpg ở bất kỳ thư mục nào
            output_images = glob.glob(f"**/output_{image_id}.jpg", recursive=True)

            if output_images:
                latest_result = max(output_images, key=os.path.getmtime)  # Lấy ảnh mới nhất nếu có nhiều ảnh trùng tên
                self.display_image(latest_result)
                self.label.config(text="✅ Hoàn tất! Đây là kết quả.")
            else:
                print("📂 Danh sách file trong thư mục:", os.listdir(image_dir))  # Debug kiểm tra
                self.label.config(text="❌ Không tìm thấy ảnh kết quả!")
        except Exception as e:
            self.label.config(text=f"❌ Lỗi: {e}")
            print(f"❌ Lỗi xảy ra: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
