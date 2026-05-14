import os
import shutil
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import datetime

class DatasetMergerTab(tk.Frame):
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.dataset_list = []
        self.setup_ui()

    def setup_ui(self):
        padding = {'padx': 20, 'pady': 5}
        
        # 头部说明
        tk.Label(self, text="📁 多数据集合并 (推荐：高精度路径)", font=("微软雅黑", 12, "bold"), fg="#2E7D32").pack(anchor="w", **padding)
        tk.Label(self, text="说明：将多个项目的数据集合并，自动处理类别 ID 偏移，生成可直接训练的新数据集项目。", fg="gray").pack(anchor="w", padx=20)

        # 列表区域
        list_frame = tk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.listbox = tk.Listbox(list_frame, height=8, font=("Consolas", 10))
        self.listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        # 按钮区域
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Button(btn_frame, text="➕ 添加数据集 (data.yaml)", command=self.add_dataset).pack(side="left", padx=5)
        tk.Button(btn_frame, text="➖ 移除选中", command=self.remove_selected).pack(side="left", padx=5)
        tk.Button(btn_frame, text="🧹 清空列表", command=self.clear_list).pack(side="left", padx=5)

        # 状态与执行
        self.status_label = tk.Label(self, text="就绪", fg="blue")
        self.status_label.pack(pady=5)
        
        tk.Button(self, text="🔗 开始合并数据集", bg="#4CAF50", fg="white", font=("微软雅黑", 11, "bold"), 
                  command=self.run_merge).pack(pady=10, ipadx=40)

        # 日志输出
        self.log_text = scrolledtext.ScrolledText(self, height=10, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.log_text.pack(fill="x", padx=20, pady=10)

    def log(self, message, color=None):
        self.log_text.insert(tk.END, message + "\n")
        if color:
            # 简单实现，实际可扩展 tag
            pass
        self.log_text.see(tk.END)

    def add_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if file_path:
            if file_path not in self.dataset_list:
                self.dataset_list.append(file_path)
                self.listbox.insert(tk.END, f"{os.path.basename(os.path.dirname(file_path))} -> {file_path}")
            else:
                messagebox.showwarning("提示", "该数据集已在列表中")

    def remove_selected(self):
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            self.listbox.delete(index)
            self.dataset_list.pop(index)

    def clear_list(self):
        self.listbox.delete(0, tk.END)
        self.dataset_list = []

    def run_merge(self):
        if len(self.dataset_list) < 2:
            messagebox.showerror("错误", "至少需要两个数据集才能合并")
            return
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_root = os.path.join("dataset", f"Merged_Project_{timestamp}")
            os.makedirs(output_root, exist_ok=True)
            
            merged_images_dir = os.path.join(output_root, "images")
            merged_labels_dir = os.path.join(output_root, "labels")
            
            os.makedirs(os.path.join(merged_images_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(merged_labels_dir, "train"), exist_ok=True)
            
            all_names = []
            id_offset = 0
            
            self.log(f"--- 开始合并任务: {timestamp} ---")
            
            for idx, yaml_path in enumerate(self.dataset_list):
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                names = data.get('names', [])
                if isinstance(names, dict):
                    names = list(names.values())
                
                self.log(f"处理数据集 {idx+1}: {os.path.basename(yaml_path)} (类别数: {len(names)})")
                
                # 确定当前数据集的根目录
                ds_root = os.path.dirname(yaml_path)
                
                # 处理训练集文件 (为了简化，这里只演示 train，val 可类比)
                train_subpath = data.get('train', 'train/images')
                # 常见结构可能是 images/train 或 train/images
                # 我们假设标准 yolo 结构
                src_img_dir = os.path.join(ds_root, train_subpath) if not os.path.isabs(train_subpath) else train_subpath
                
                if not os.path.exists(src_img_dir):
                    # 尝试寻找兄弟目录
                    src_img_dir = os.path.join(ds_root, "images", "train")
                
                if os.path.exists(src_img_dir):
                    for img_name in os.listdir(src_img_dir):
                        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                            # 复制图片
                            src_img = os.path.join(src_img_dir, img_name)
                            # 为了防止重名，添加数据集索引前缀
                            new_img_name = f"ds{idx}_{img_name}"
                            dst_img = os.path.join(merged_images_dir, "train", new_img_name)
                            shutil.copy2(src_img, dst_img)
                            
                            # 处理对应的 label
                            label_name = os.path.splitext(img_name)[0] + ".txt"
                            # 寻找 label 目录
                            src_label_dir = src_img_dir.replace("images", "labels")
                            src_label = os.path.join(src_label_dir, label_name)
                            
                            if os.path.exists(src_label):
                                dst_label = os.path.join(merged_labels_dir, "train", os.path.splitext(new_img_name)[0] + ".txt")
                                # 读取并偏移 ID
                                with open(src_label, 'r') as lf:
                                    lines = lf.readlines()
                                
                                new_lines = []
                                for line in lines:
                                    parts = line.strip().split()
                                    if parts:
                                        old_id = int(parts[0])
                                        new_id = old_id + id_offset
                                        new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                                
                                with open(dst_label, 'w') as nlf:
                                    nlf.writelines(new_lines)

                all_names.extend(names)
                id_offset += len(names)

            # 生成新的 data.yaml
            new_yaml_data = {
                'path': output_root,
                'train': 'images/train',
                'val': 'images/train', # 简化，默认全用
                'names': {i: name for i, name in enumerate(all_names)}
            }
            
            with open(os.path.join(output_root, "data.yaml"), 'w', encoding='utf-8') as f:
                yaml.dump(new_yaml_data, f, allow_unicode=True)

            self.log(f"✅ 合并成功！")
            self.log(f"输出目录: {output_root}")
            self.log(f"总类别数: {len(all_names)}")
            self.status_label.config(text="合并成功", fg="green")
            messagebox.showinfo("成功", f"数据集已合并至:\n{output_root}\n\n您可以直接使用该目录下的 data.yaml 进行新模型的训练。")

        except Exception as e:
            self.log(f"❌ 出错: {str(e)}")
            messagebox.showerror("错误", f"合并过程中出错: {str(e)}")
            self.status_label.config(text="合并失败", fg="red")
