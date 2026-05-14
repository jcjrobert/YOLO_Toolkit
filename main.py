import tkinter as tk
from tkinter import ttk
from config_manager import ConfigManager
from modules.tab_image_resizer import ImageResizerTab
from modules.tab_dataset_split import DatasetSplitterTab
from modules.tab_model_train import ModelTrainTab
from modules.tab_model_infer import ModelInferTab
from modules.tab_dataset_merge import DatasetMergerTab
from modules.tab_model_merge import ModelMergeTab

class YOLO_Toolkit_App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 多模态综合工具箱")
        self.root.geometry("900x750")

        # 初始化配置管理器
        self.config_manager = ConfigManager()

        # 创建 Notebook (Tab 容器)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # 实例化各个 Tab
        self.tab1 = ImageResizerTab(self.notebook, self.config_manager)
        self.tab2 = DatasetSplitterTab(self.notebook, self.config_manager)
        self.tab3 = ModelTrainTab(self.notebook, self.config_manager)
        self.tab4 = ModelInferTab(self.notebook, self.config_manager)
        self.tab5 = DatasetMergerTab(self.notebook, self.config_manager)
        self.tab6 = ModelMergeTab(self.notebook, self.config_manager)

        # 将 Tab 添加到 Notebook
        self.notebook.add(self.tab1, text=" 1. 图像整理 ")
        self.notebook.add(self.tab2, text=" 2. 数据集分类 ")
        self.notebook.add(self.tab3, text=" 3. 模型训练 ")
        self.notebook.add(self.tab4, text=" 4. 模型推理 ")
        self.notebook.add(self.tab5, text=" 5. 数据集合并 ")
        self.notebook.add(self.tab6, text=" 6. 模型物理合并 ")

        # 绑定窗口关闭事件以保存最终配置
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        # 可以在这里做最后的配置同步
        self.config_manager.save_config()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    
    # 设置样式
    style = ttk.Style()
    style.configure("TNotebook.Tab", font=("微软雅黑", 10))
    
    app = YOLO_Toolkit_App(root)
    root.mainloop()
