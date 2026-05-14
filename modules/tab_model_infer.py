import os
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import cv2
from ultralytics import YOLO

class ModelInferTab(tk.Frame):
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()

    def setup_ui(self):
        padding = {'padx': 20, 'pady': 10}
        
        # 模型选择
        tk.Label(self, text="选择 YOLO 模型 (.pt):").pack(anchor="w", **padding)
        frame_model = tk.Frame(self)
        frame_model.pack(fill="x", padx=20)
        self.model_path = tk.StringVar(value=self.config_manager.get("yolo_infer", "model_path"))
        tk.Entry(frame_model, textvariable=self.model_path).pack(side="left", fill="x", expand=True)
        tk.Button(frame_model, text="浏览", command=self.browse_model).pack(side="right", padx=5)

        # 输入选择
        tk.Label(self, text="输入图片或视频:").pack(anchor="w", **padding)
        frame_input = tk.Frame(self)
        frame_input.pack(fill="x", padx=20)
        self.input_path = tk.StringVar(value=self.config_manager.get("yolo_infer", "input_path"))
        tk.Entry(frame_input, textvariable=self.input_path).pack(side="left", fill="x", expand=True)
        tk.Button(frame_input, text="选择", command=self.browse_input).pack(side="right", padx=5)

        # 输出目录
        tk.Label(self, text="结果保存目录:").pack(anchor="w", **padding)
        self.output_dir = tk.StringVar(value=self.config_manager.get("yolo_infer", "output_dir"))
        tk.Entry(self, textvariable=self.output_dir, width=30).pack(anchor="w", padx=20)

        # 进度条
        self.progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill="x", padx=20, pady=20)

        self.status_label = tk.Label(self, text="就绪", fg="blue")
        self.status_label.pack()

        self.run_btn = tk.Button(self, text="🚀 开始执行推理", command=self.run_inference_thread, 
                                 bg="#0078D7", fg="white", font=("Arial", 12, "bold"), height=2)
        self.run_btn.pack(pady=10, ipadx=50)

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Weights", "*.pt")])
        if path:
            self.model_path.set(path)
            self.config_manager.set("yolo_infer", "model_path", path)

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Media files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv")])
        if path:
            self.input_path.set(path)
            self.config_manager.set("yolo_infer", "input_path", path)

    def log(self, text):
        self.status_label.config(text=text)
        self.update_idletasks()

    def run_inference_thread(self):
        if not self.model_path.get() or not self.input_path.get():
            messagebox.showwarning("提示", "请选择模型和输入文件")
            return
        
        self.config_manager.set("yolo_infer", "output_dir", self.output_dir.get())
        
        self.run_btn.config(state="disabled", text="正在处理...")
        thread = threading.Thread(target=self.run_inference)
        thread.daemon = True
        thread.start()

    def run_inference(self):
        input_path = os.path.normpath(self.input_path.get())
        model_path = os.path.normpath(self.model_path.get())
        out_dir = self.output_dir.get()
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        try:
            self.log("正在加载模型...")
            model = YOLO(model_path)
            
            ext = os.path.splitext(input_path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.log(f"正在处理图片: {os.path.basename(input_path)}")
                model.predict(source=input_path, save=True, project=out_dir, name='predict', exist_ok=True)
                self.progress['value'] = 100
            else:
                self.process_video(model, input_path, out_dir)
                
            self.log(f"推理完成！保存至: {out_dir}")
            messagebox.showinfo("成功", f"处理完成！\n结果保存至: {os.path.abspath(out_dir)}")
        except Exception as e:
            self.log(f"失败: {str(e)}")
            messagebox.showerror("错误", str(e))
        finally:
            self.run_btn.config(state="normal", text="🚀 开始执行推理")
            self.progress['value'] = 0

    def process_video(self, model, input_path, out_dir):
        tmp_dir = "temp_frames_raw"
        tmp_out = "temp_frames_annotated"
        for d in [tmp_dir, tmp_out]:
            if os.path.exists(d): shutil.rmtree(d)
            os.makedirs(d)
        
        self.log("Step 1/3: 提取帧...")
        subprocess.call(f'ffmpeg -i "{input_path}" -q:v 2 -start_number 0 "{tmp_dir}/%05d.jpg"', shell=True)
        
        raw_frames = sorted([f for f in os.listdir(tmp_dir) if f.endswith('.jpg')])
        total_frames = len(raw_frames)
        if total_frames == 0: raise Exception("FFmpeg 提取帧失败")

        self.log(f"Step 2/3: YOLO 推理 (共 {total_frames} 帧)...")
        for i, frame_name in enumerate(raw_frames):
            frame_path = os.path.join(tmp_dir, frame_name)
            results = model.predict(frame_path, verbose=False)
            annotated_frame = results[0].plot()
            cv2.imwrite(os.path.join(tmp_out, frame_name), annotated_frame)
            self.progress['value'] = int(((i + 1) / total_frames) * 100)
            if (i+1) % 10 == 0: self.log(f"已处理: {i+1}/{total_frames}")

        self.log("Step 3/3: 合成视频...")
        output_filename = os.path.splitext(os.path.basename(input_path))[0] + "_result.mp4"
        output_path = os.path.join(out_dir, output_filename)
        
        # 尝试获取 FPS
        fps = 25.0
        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        except: pass

        subprocess.call(f'ffmpeg -y -r {fps} -i "{tmp_out}/%05d.jpg" -c:v libx264 -pix_fmt yuv420p "{output_path}"', shell=True)
        
        shutil.rmtree(tmp_dir)
        shutil.rmtree(tmp_out)
