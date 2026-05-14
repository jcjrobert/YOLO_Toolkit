import os
import random
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import yaml

class DatasetSplitterTab(tk.Frame):
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()

    def setup_ui(self):
        padding = {'padx': 20, 'pady': 10}
        
        tk.Label(self, text="Label-Studio 导出解压文件夹:").pack(anchor="w", **padding)
        frame_src = tk.Frame(self)
        frame_src.pack(fill="x", padx=20)
        self.src_dir_var = tk.StringVar(value=self.config_manager.get("dataset_splitter", "src_dir"))
        tk.Entry(frame_src, textvariable=self.src_dir_var).pack(side="left", fill="x", expand=True)
        tk.Button(frame_src, text="选择", command=self.browse_src).pack(side="right", padx=5)

        # 比例设置
        ratio_frame = tk.LabelFrame(self, text="数据集切分比例 (总和应为 1.0)", padx=10, pady=10)
        ratio_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(ratio_frame, text="训练 (Train):").grid(row=0, column=0)
        self.train_ratio = tk.StringVar(value=str(self.config_manager.get("dataset_splitter", "train_ratio")))
        tk.Entry(ratio_frame, textvariable=self.train_ratio, width=8).grid(row=0, column=1, padx=5)

        tk.Label(ratio_frame, text="验证 (Val):").grid(row=0, column=2)
        self.val_ratio = tk.StringVar(value=str(self.config_manager.get("dataset_splitter", "val_ratio")))
        tk.Entry(ratio_frame, textvariable=self.val_ratio, width=8).grid(row=0, column=3, padx=5)

        tk.Label(ratio_frame, text="测试 (Test):").grid(row=0, column=4)
        self.test_ratio = tk.StringVar(value=str(self.config_manager.get("dataset_splitter", "test_ratio")))
        tk.Entry(ratio_frame, textvariable=self.test_ratio, width=8).grid(row=0, column=5, padx=5)

        self.status_label = tk.Label(self, text="准备就绪", fg="blue")
        self.status_label.pack(pady=20)

        tk.Button(self, text="🚀 开始切分数据集并生成 data.yaml", bg="#2196F3", fg="white", 
                  font=("Arial", 12, "bold"), command=self.run_split).pack(pady=10, ipadx=30)

    def browse_src(self):
        directory = filedialog.askdirectory()
        if directory:
            self.src_dir_var.set(directory)
            self.config_manager.set("dataset_splitter", "src_dir", directory)

    def run_split(self):
        src_path = self.src_dir_var.get()
        if not src_path or not os.path.exists(src_path):
            messagebox.showerror("错误", "无效的文件夹路径")
            return

        try:
            r_train = float(self.train_ratio.get())
            r_val = float(self.val_ratio.get())
            r_test = float(self.test_ratio.get())
        except ValueError:
            messagebox.showerror("错误", "比例必须是数字")
            return

        # 保存配置
        self.config_manager.set("dataset_splitter", "train_ratio", r_train)
        self.config_manager.set("dataset_splitter", "val_ratio", r_val)
        self.config_manager.set("dataset_splitter", "test_ratio", r_test)

        images_src = os.path.join(src_path, "images")
        labels_src = os.path.join(src_path, "labels")
        classes_file = os.path.join(src_path, "classes.txt")
        
        if not (os.path.exists(images_src) and os.path.exists(labels_src)):
            messagebox.showerror("错误", "未找到 images 或 labels 目录")
            return

        output_root = os.path.join(src_path, "YOLO_Dataset")
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
            os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)

        all_images = [f for f in os.listdir(images_src) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.seed(42)
        random.shuffle(all_images)

        total = len(all_images)
        train_end = int(total * r_train)
        val_end = int(total * (r_train + r_val))

        dataset_splits = {
            'train': all_images[:train_end],
            'val': all_images[train_end:val_end],
            'test': all_images[val_end:]
        }

        self.status_label.config(text="正在搬运文件...")
        self.update_idletasks()

        for split, images in dataset_splits.items():
            for img_name in images:
                shutil.copy(os.path.join(images_src, img_name), 
                            os.path.join(output_root, "images", split, img_name))
                label_name = os.path.splitext(img_name)[0] + ".txt"
                label_p = os.path.join(labels_src, label_name)
                if os.path.exists(label_p):
                    shutil.copy(label_p, os.path.join(output_root, "labels", split, label_name))

        class_names = []
        if os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
        
        yaml_data = {
            'path': os.path.abspath(output_root),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {i: name for i, name in enumerate(class_names)}
        }

        yaml_path = os.path.join(output_root, "data.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

        self.status_label.config(text="切分完成！", fg="green")
        messagebox.showinfo("成功", f"处理完成！\n数据集已保存至: {output_root}\ndata.yaml 已生成。")
