import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from ultralytics import YOLO
import copy
import datetime

class ModelMergeTab(tk.Frame):
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.model_list = []
        self.setup_ui()

    def setup_ui(self):
        padding = {'padx': 20, 'pady': 5}
        
        # 头部说明
        tk.Label(self, text="⚡ 多模型物理合并 (快捷路径：免重新训练)", font=("微软雅黑", 12, "bold"), fg="#1976D2").pack(anchor="w", **padding)
        tk.Label(self, text="说明：通过拼接检测头权重，将多个识别不同物体的模型合并为一个文件。要求所有模型架构必须一致。", fg="gray").pack(anchor="w", padx=20)

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
        
        tk.Button(btn_frame, text="➕ 添加模型 (.pt)", command=self.add_model).pack(side="left", padx=5)
        tk.Button(btn_frame, text="➖ 移除选中", command=self.remove_selected).pack(side="left", padx=5)
        
        # 执行区域
        tk.Button(self, text="🧠 执行物理权重拼接", bg="#2196F3", fg="white", font=("微软雅黑", 11, "bold"), 
                  command=self.run_merge).pack(pady=10, ipadx=40)

        # 日志输出
        self.log_text = scrolledtext.ScrolledText(self, height=12, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=20, pady=10)

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def add_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch files", "*.pt")])
        if file_path:
            if file_path not in self.model_list:
                self.model_list.append(file_path)
                self.listbox.insert(tk.END, f"{os.path.basename(file_path)} -> {file_path}")
            else:
                messagebox.showwarning("提示", "该模型已在列表中")

    def remove_selected(self):
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            self.listbox.delete(index)
            self.model_list.pop(index)

    def run_merge(self):
        if len(self.model_list) < 2:
            messagebox.showerror("错误", "至少需要两个模型才能合并")
            return
        
        self.log(f"--- 模型物理合并任务启动: {datetime.datetime.now()} ---")
        
        try:
            # 1. 加载所有模型字典
            ckpt_list = []
            for p in self.model_list:
                self.log(f"正在加载: {os.path.basename(p)}")
                ckpt = torch.load(p, map_location='cpu')
                ckpt_list.append(ckpt)

            # 2. 以第一个模型为基础
            merged_ckpt = ckpt_list[0] # 这里深拷贝更好，但 torch.load 已经是新对象了
            base_model = merged_ckpt['model']
            
            # 提取标签和类别数
            all_names = merged_ckpt.get('metadata', {}).get('names', merged_ckpt.get('names', {}))
            if isinstance(all_names, list):
                all_names = {i: n for i, n in enumerate(all_names)}
            
            # 3. 寻找检测层 (Detect 层在 YOLO v8/v11 中通常是最后一层)
            # 我们需要拼接的是 Detect.cv3 (类别预测) 的权重和偏置
            # 这是一个非常有挑战性的操作，因为不同版本的层索引可能不同
            # 这里采用名称匹配策略
            
            self.log("正在执行检测头权重拼接...")
            
            current_nc = len(all_names)
            
            for i in range(1, len(ckpt_list)):
                ext_ckpt = ckpt_list[i]
                ext_names = ext_ckpt.get('metadata', {}).get('names', ext_ckpt.get('names', {}))
                if isinstance(ext_names, list):
                    ext_names = {idx: n for idx, n in enumerate(ext_names)}
                
                ext_nc = len(ext_names)
                self.log(f"合并扩展模型: {os.path.basename(self.model_list[i])} (类别数: {ext_nc})")
                
                # 更新名称字典
                for idx, name in ext_names.items():
                    all_names[current_nc + idx] = name
                
                # 拼接权重 (State Dict 手术)
                # YOLOv8/v11 的 Detect 层通常包含 cv2 (box) 和 cv3 (cls)
                # 我们主要合并 cv3 的输出通道
                state_dict = base_model.state_dict()
                ext_state_dict = ext_ckpt['model'].state_dict()
                
                for key in state_dict.keys():
                    if '.cv3.' in key and ('.weight' in key or '.bias' in key):
                        # 找到对应的类别预测层
                        # 输出通道维度：cv3[i].weight 的 shape 是 [nc, input, 1, 1]
                        w_base = state_dict[key]
                        w_ext = ext_state_dict[key]
                        
                        # 执行拼接
                        # 注意：YOLO 的 Detect 层 output = [box_coords (4) + nc]
                        # 实际上 v8 分开了 cv2 (box) 和 cv3 (cls)
                        # 所以直接拼接 cv3 的输出维度即可
                        state_dict[key] = torch.cat([w_base, w_ext], dim=0)
                
                base_model.load_state_dict(state_dict, strict=False)
                current_nc += ext_nc

            # 4. 更新元数据
            merged_ckpt['model'] = base_model
            merged_ckpt['nc'] = current_nc
            if 'metadata' in merged_ckpt:
                merged_ckpt['metadata']['names'] = all_names
                merged_ckpt['metadata']['nc'] = current_nc
            else:
                merged_ckpt['names'] = all_names
            
            # 5. 保存
            os.makedirs("models", exist_ok=True)
            out_name = f"physically_merged_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            out_path = os.path.join("models", out_name)
            
            # 这里的保存需要注意，ultralytics 加载时会对模型结构进行校验
            # 物理拼接可能导致结构不匹配校验，因此这种方法主要用于特定场景
            torch.save(merged_ckpt, out_path)
            
            self.log(f"\n✅ 物理合并完成！")
            self.log(f"保存路径: {out_path}")
            self.log(f"最终类别数: {current_nc}")
            self.log(f"最终标签: {list(all_names.values())}")
            
            messagebox.showinfo("成功", f"模型已物理合并至:\n{out_path}\n\n注意：物理合并的模型建议先在推理 Tab 中测试效果。")

        except Exception as e:
            self.log(f"❌ 拼接失败: {str(e)}")
            messagebox.showerror("错误", f"物理合并失败: {str(e)}")
