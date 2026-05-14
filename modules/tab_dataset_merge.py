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
            self.log("--- 开始合并任务 ---")
            
            # 1. 预读所有数据集，确定类别和输出目录名
            dataset_info = []
            unique_classes = []
            
            for yaml_path in self.dataset_list:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                names = data.get('names', [])
                if isinstance(names, dict):
                    names = [names[i] for i in sorted(names.keys())]
                
                dataset_info.append({
                    'path': yaml_path,
                    'data': data,
                    'names': names,
                    'root': os.path.dirname(yaml_path)
                })
                
                for name in names:
                    if name not in unique_classes:
                        unique_classes.append(name)
            
            # 2. 生成文件夹名称: [类别1]_[类别2]_[时间戳] (YYYYMMDDHHmmss)
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            classes_str = "_".join(unique_classes)
            folder_name = f"{classes_str}_{timestamp}"
            output_root = os.path.join("dataset", folder_name)
            
            os.makedirs(output_root, exist_ok=True)
            self.log(f"创建输出目录: {output_root}")

            # 3. 执行合并
            merged_names = []
            id_offset = 0
            subsets = ['train', 'val', 'test']
            
            for idx, info in enumerate(dataset_info):
                yaml_path = info['path']
                data = info['data']
                names = info['names']
                ds_root = info['root']
                
                self.log(f"正在处理数据集 ({idx+1}/{len(dataset_info)}): {os.path.basename(yaml_path)}")
                
                for subset in subsets:
                    # 获取子集路径 (优先从 yaml 读取)
                    subset_path = data.get(subset)
                    if not subset_path:
                        # 尝试猜测默认路径
                        possible_paths = [
                            os.path.join("images", subset),
                            os.path.join(subset, "images"),
                            subset
                        ]
                        for p in possible_paths:
                            full_p = os.path.join(ds_root, p)
                            if os.path.exists(full_p):
                                subset_path = p
                                break
                    
                    if not subset_path:
                        continue

                    src_img_dir = os.path.join(ds_root, subset_path) if not os.path.isabs(subset_path) else subset_path
                    if not os.path.exists(src_img_dir):
                        continue

                    # 创建对应的目标目录
                    dst_img_dir = os.path.join(output_root, "images", subset)
                    dst_lbl_dir = os.path.join(output_root, "labels", subset)
                    os.makedirs(dst_img_dir, exist_ok=True)
                    os.makedirs(dst_lbl_dir, exist_ok=True)

                    count = 0
                    for img_name in os.listdir(src_img_dir):
                        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                            # 复制图片 (添加前缀防止冲突)
                            src_img_path = os.path.join(src_img_dir, img_name)
                            new_img_name = f"ds{idx}_{img_name}"
                            dst_img_path = os.path.join(dst_img_dir, new_img_name)
                            shutil.copy2(src_img_path, dst_img_path)
                            
                            # 处理标签
                            label_name = os.path.splitext(img_name)[0] + ".txt"
                            # 假设标准 YOLO 结构: images -> labels
                            src_lbl_dir = src_img_dir.replace("images", "labels")
                            src_lbl_path = os.path.join(src_lbl_dir, label_name)
                            
                            if os.path.exists(src_lbl_path):
                                dst_lbl_path = os.path.join(dst_lbl_dir, os.path.splitext(new_img_name)[0] + ".txt")
                                with open(src_lbl_path, 'r', encoding='utf-8') as lf:
                                    lines = lf.readlines()
                                
                                new_lines = []
                                for line in lines:
                                    parts = line.strip().split()
                                    if parts:
                                        old_id = int(parts[0])
                                        new_id = old_id + id_offset
                                        new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                                
                                with open(dst_lbl_path, 'w', encoding='utf-8') as nlf:
                                    nlf.writelines(new_lines)
                            count += 1
                    
                    if count > 0:
                        self.log(f"  - 子集 '{subset}': 已合并 {count} 张图片")

                merged_names.extend(names)
                id_offset += len(names)

            # 4. 生成新的 data.yaml
            new_yaml_data = {
                'path': os.path.abspath(output_root),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'names': {i: name for i, name in enumerate(merged_names)}
            }
            
            # 清理不存在的路径
            for key in ['val', 'test']:
                if not os.path.exists(os.path.join(output_root, "images", key)):
                    del new_yaml_data[key]

            with open(os.path.join(output_root, "data.yaml"), 'w', encoding='utf-8') as f:
                yaml.dump(new_yaml_data, f, allow_unicode=True, sort_keys=False)

            self.log(f"✅ 合并成功！")
            self.log(f"输出目录: {output_root}")
            self.log(f"总类别数: {len(merged_names)}")
            self.status_label.config(text="合并成功", fg="green")
            messagebox.showinfo("成功", f"数据集已合并至:\n{output_root}")

        except Exception as e:
            self.log(f"❌ 出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("错误", f"合并过程中出错: {str(e)}")
            self.status_label.config(text="合并失败", fg="red")

        except Exception as e:
            self.log(f"❌ 出错: {str(e)}")
            messagebox.showerror("错误", f"合并过程中出错: {str(e)}")
            self.status_label.config(text="合并失败", fg="red")
