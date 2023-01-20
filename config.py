import yaml
import os


def load_config():
    current_path = os.path.abspath(".")
    yaml_path = os.path.join(current_path, "config.yaml")
    # 打开yaml文件
    print("***获取yaml文件数据***")
    file = open(yaml_path, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()

    print(file_data)
    data = yaml.load(file_data)
    return data



t = load_config()
pass
