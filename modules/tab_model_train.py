import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image
from ultralytics import YOLO
import datetime
import shutil
import yaml
import re

class TextLogger:
    """重定向输出到 Tkinter Text 控件和文件"""
    def __init__(self, text_widget, log_file=None):
        self.text_widget = text_widget
        self.log_file = log_file
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        if self.text_widget:
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(message)

    def flush(self):
        self.terminal.flush()

class ModelTrainTab(tk.Frame):
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()

    def setup_ui(self):
        padding = {'padx': 20, 'pady': 5}
        
        # 顶部容器
        top_frame = tk.Frame(self)
        top_frame.pack(fill="x")

        # 1. 选择数据配置
        tk.Label(top_frame, text="数据集配置 (data.yaml):").pack(anchor="w", **padding)
        frame_yaml = tk.Frame(top_frame)
        frame_yaml.pack(fill="x", padx=20)
        self.yaml_path = tk.StringVar(value=self.config_manager.get("yolo_train", "yaml_path"))
        tk.Entry(frame_yaml, textvariable=self.yaml_path).pack(side="left", fill="x", expand=True)
        tk.Button(frame_yaml, text="浏览", command=self.browse_yaml).pack(side="right", padx=5)

        # 2. 选择权重文件
        tk.Label(top_frame, text="预训练模型 (weights):").pack(anchor="w", **padding)
        frame_model = tk.Frame(top_frame)
        frame_model.pack(fill="x", padx=20)
        self.model_path = tk.StringVar(value=self.config_manager.get("yolo_train", "model_path"))
        tk.Entry(frame_model, textvariable=self.model_path).pack(side="left", fill="x", expand=True)
        tk.Button(frame_model, text="浏览", command=self.browse_model).pack(side="right", padx=5)

        # 3. 训练参数设置
        params_frame = tk.LabelFrame(top_frame, text="训练参数设置", padx=10, pady=10)
        params_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky="e")
        self.epochs = tk.StringVar(value=str(self.config_manager.get("yolo_train", "epochs")))
        tk.Entry(params_frame, textvariable=self.epochs, width=10).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, sticky="e")
        self.batch = tk.StringVar(value=str(self.config_manager.get("yolo_train", "batch")))
        tk.Entry(params_frame, textvariable=self.batch, width=10).grid(row=0, column=3, padx=5, pady=5)

        tk.Label(params_frame, text="Device:").grid(row=1, column=0, sticky="e")
        self.device = tk.StringVar(value=self.config_manager.get("yolo_train", "device"))
        tk.Entry(params_frame, textvariable=self.device, width=10).grid(row=1, column=1, padx=5, pady=5)

        # 模型融合入口预留
        tk.Button(params_frame, text="🧩 模型融合(开发中)", state="disabled").grid(row=1, column=2, columnspan=2, padx=5, pady=5)

        # 4. 控制区域
        ctrl_frame = tk.Frame(top_frame)
        ctrl_frame.pack(fill="x", pady=10)
        
        self.start_btn = tk.Button(ctrl_frame, text="🚀 开始训练", bg="#4CAF50", fg="white", 
                                   font=("Arial", 11, "bold"), command=self.start_training_thread)
        self.start_btn.pack(side="left", padx=20, expand=True, fill="x")

        self.status_label = tk.Label(ctrl_frame, text="等待操作", fg="blue")
        self.status_label.pack(side="right", padx=20)

        # 5. 日志文本框
        tk.Label(self, text="训练日志:").pack(anchor="w", padx=20)
        self.log_text = scrolledtext.ScrolledText(self, height=15, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True, padx=20, pady=5)

    def browse_yaml(self):
        path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if path:
            self.yaml_path.set(path)
            self.config_manager.set("yolo_train", "yaml_path", path)

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
        if path:
            self.model_path.set(path)
            self.config_manager.set("yolo_train", "model_path", path)

    def clean_tags(self, tags):
        """清洗标签名，去除不支持的字符"""
        # 合并成字符串
        tag_str = "_".join(tags)
        # 只保留字母数字和下划线
        return re.sub(r'[^\w\s-]', '', tag_str).replace(' ', '_')

    def get_dataset_info(self, yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            names = data.get('names', [])
            if isinstance(names, dict):
                tags = list(names.values())
            else:
                tags = names
            return tags
        except:
            return ["unknown"]

    def start_training_thread(self):
        if not self.yaml_path.get() or not os.path.exists(self.yaml_path.get()):
            self.log_text.insert(tk.END, "❌ 错误: 请先选择有效的 data.yaml 文件\n", "error")
            self.log_text.tag_config("error", foreground="red")
            return
        
        # 保存参数
        self.config_manager.set("yolo_train", "epochs", int(self.epochs.get()))
        self.config_manager.set("yolo_train", "batch", int(self.batch.get()))
        self.config_manager.set("yolo_train", "device", self.device.get())

        self.start_btn.config(state="disabled", text="训练中...")
        self.log_text.delete(1.0, tk.END)
        
        t = threading.Thread(target=self.run_train)
        t.daemon = True
        t.start()

    def run_train(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_file_path = None
        
        try:
            yaml_p = self.yaml_path.get()
            model_p = self.model_path.get()
            
            # 1. 动态命名逻辑
            tags = self.get_dataset_info(yaml_p)
            tag_name_clean = self.clean_tags(tags)
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            
            project_dir_name = f"yolo_project_{date_str}_{tag_name_clean}"
            project_root = self.config_manager.get("yolo_train", "project_root") or "project"
            save_dir = os.path.join(project_root, project_dir_name)
            
            # 2. 初始化日志
            os.makedirs("logs", exist_ok=True)
            log_file_path = os.path.join("logs", f"train_{date_str}_{tag_name_clean}.log")
            
            logger = TextLogger(self.log_text, log_file_path)
            sys.stdout = logger
            sys.stderr = logger
            
            self.status_label.config(text="状态: 初始化模型...", fg="orange")
            print(f"--- 训练开始: {datetime.datetime.now()} ---")
            print(f"数据集: {yaml_p}")
            print(f"标签: {tags}")
            print(f"保存目录: {save_dir}")
            
            # 3. 执行训练
            model = YOLO(model_p)
            model.train(
                data=yaml_p,
                epochs=int(self.epochs.get()),
                batch=int(self.batch.get()),
                device=self.device.get(),
                project=project_root,
                name=project_dir_name,
                exist_ok=True,
                amp=True
            )
            
            # 4. 后处理：模型复制与归档
            self.status_label.config(text="状态: 归档模型中...", fg="blue")
            best_model_path = os.path.join(save_dir, "weights", "best.pt")
            
            if os.path.exists(best_model_path):
                models_dir = "models"
                os.makedirs(models_dir, exist_ok=True)
                target_model_name = f"{tag_name_clean}_{date_str}.pt"
                target_model_path = os.path.join(models_dir, target_model_name)
                
                shutil.copy(best_model_path, target_model_path)
                print(f"\n✅ 模型已提取并归档至: {target_model_path}")
                
                # 5. 生成 Readme
                readme_path = os.path.join(save_dir, "readme.txt")
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(f"训练日期: {date_str}\n")
                    f.write(f"标签列表: {', '.join(tags)}\n")
                    f.write(f"原始数据集: {yaml_p}\n")
                    f.write(f"基础模型: {model_p}\n")
                    f.write(f"训练参数: Epochs={self.epochs.get()}, Batch={self.batch.get()}, Device={self.device.get()}\n")
                print(f"✅ 项目说明文档已生成: {readme_path}")
            else:
                print("\n⚠️ 警告: 未找到训练后的 best.pt 模型文件。")

            self.status_label.config(text="状态: 训练成功", fg="green")
            
        except Exception as e:
            print(f"\n❌ 训练出错:\n{str(e)}")
            self.status_label.config(text="状态: 训练出错", fg="red")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.start_btn.config(state="normal", text="🚀 开始训练")
