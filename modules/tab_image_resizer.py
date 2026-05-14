import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image

class ImageResizerTab(tk.Frame):
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()

    def setup_ui(self):
        padding = {'padx': 20, 'pady': 10}
        
        # 输入目录
        tk.Label(self, text="输入图片文件夹:").pack(anchor="w", **padding)
        frame_input = tk.Frame(self)
        frame_input.pack(fill="x", padx=20)
        self.input_dir_var = tk.StringVar(value=self.config_manager.get("image_resizer", "input_dir"))
        tk.Entry(frame_input, textvariable=self.input_dir_var).pack(side="left", fill="x", expand=True)
        tk.Button(frame_input, text="选择", command=self.browse_input).pack(side="right", padx=5)

        # 目标尺寸
        tk.Label(self, text="目标长边尺寸 (像素):").pack(anchor="w", **padding)
        self.target_size_var = tk.StringVar(value=str(self.config_manager.get("image_resizer", "target_size")))
        tk.Entry(self, textvariable=self.target_size_var, width=10).pack(anchor="w", padx=20)

        # 输出目录
        tk.Label(self, text="输出文件夹名称:").pack(anchor="w", **padding)
        self.output_dir_var = tk.StringVar(value=self.config_manager.get("image_resizer", "output_dir"))
        tk.Entry(self, textvariable=self.output_dir_var, width=30).pack(anchor="w", padx=20)

        # 状态
        self.status_label = tk.Label(self, text="就绪", fg="blue")
        self.status_label.pack(pady=20)

        # 开始按钮
        tk.Button(self, text="🚀 开始批量缩小并转换 (PNG->JPG)", bg="#4CAF50", fg="white", 
                  font=("Arial", 12, "bold"), command=self.run_resizer).pack(pady=10, ipadx=50)

    def browse_input(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir_var.set(directory)
            self.config_manager.set("image_resizer", "input_dir", directory)

    def run_resizer(self):
        input_folder = self.input_dir_var.get()
        try:
            target_size = int(self.target_size_var.get())
        except ValueError:
            messagebox.showerror("错误", "尺寸必须是整数")
            return
        output_folder_name = self.output_dir_var.get()
        
        if not input_folder or not os.path.exists(input_folder):
            messagebox.showerror("错误", "请输入有效的输入目录")
            return

        # 保存配置
        self.config_manager.set("image_resizer", "target_size", target_size)
        self.config_manager.set("image_resizer", "output_dir", output_folder_name)

        output_folder = os.path.join(input_folder, output_folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not files:
            messagebox.showinfo("提示", "未找到图片文件")
            return

        count = 0
        for file_name in files:
            try:
                img_path = os.path.join(input_folder, file_name)
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w > h:
                        new_w = target_size
                        new_h = int(h * (target_size / w))
                    else:
                        new_h = target_size
                        new_w = int(w * (target_size / h))
                    
                    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    if img_resized.mode in ("RGBA", "P"):
                        img_resized = img_resized.convert("RGB")
                    elif img_resized.mode != "RGB":
                        img_resized = img_resized.convert("RGB")
                    
                    base_name = os.path.splitext(file_name)[0]
                    save_path = os.path.join(output_folder, f"{base_name}.jpg")
                    img_resized.save(save_path, "JPEG", quality=95)
                    count += 1
                    self.status_label.config(text=f"正在处理: {count}/{len(files)}")
                    self.update_idletasks()
            except Exception as e:
                print(f"处理 {file_name} 出错: {e}")

        self.status_label.config(text=f"处理完成！共转换 {count} 张图片。", fg="green")
        messagebox.showinfo("成功", f"处理完成！保存至: {output_folder}")
