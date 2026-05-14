import json
import os

DEFAULT_CONFIG = {
    "image_resizer": {
        "target_size": 640,
        "input_dir": "",
        "output_dir": "resized_output"
    },
    "dataset_splitter": {
        "src_dir": "",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1
    },
    "yolo_train": {
        "yaml_path": "",
        "model_path": "yolo11n.pt",
        "epochs": 100,
        "batch": 8,
        "workers": 0,
        "device": "0",
        "project_name": "My_YOLO_Project"
    },
    "yolo_infer": {
        "model_path": "",
        "input_path": "",
        "output_dir": "inference_results"
    }
}

class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并默认配置，确保新添加的字段存在
                    full_config = DEFAULT_CONFIG.copy()
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in full_config:
                            full_config[key].update(value)
                        else:
                            full_config[key] = value
                    return full_config
            except Exception as e:
                print(f"加载配置失败: {e}")
                return DEFAULT_CONFIG.copy()
        else:
            self.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()

    def save_config(self, config=None):
        if config:
            self.config = config
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def get(self, section, key=None):
        if key:
            return self.config.get(section, {}).get(key)
        return self.config.get(section)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
