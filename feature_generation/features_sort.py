from config import load_config
from feature_generation.char_shape.simclr.models.resnet_simclr import ResNetSimCLR
import torch

class featureSort:
    def __init__(self):

        self.initialized_feature = False
        self.config = load_config()

        self.bihuashuDict = None
        self.hanzijiegouDict = None
        self.pianpangbushouDict = None
        self.sijiaobianmaDict = None

    def check_feature_initialized(self):
        if not self.initialized_feature:
            self._initialize_feature()

    def _initialize_feature(self):
        self.init_traditional_shape_feature()
        self.init_simclr_state_dict()
        self.initialized_feature = True

    def init_traditional_shape_feature(self):
        def initDict(path):
            dict = {}
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f.readlines():
                    # 移除换行符，并且根据空格拆分
                    splits = line.strip('\n').split(' ')
                    key = splits[0]
                    value = splits[1]
                    dict[key] = value
            return dict

        # 字典初始化
        self.bihuashuDict = initDict(self.config['feature']['path']['bi_hua_shu_path'])
        self.hanzijiegouDict = initDict(self.config['feature']['path']['han_zi_jie_gou_path'])
        self.pianpangbushouDict = initDict(self.config['feature']['path']['pian_pang_bu_shou_path'])
        self.sijiaobianmaDict = initDict(self.config['feature']['path']['si_jiao_bian_ma_path'])

    def init_simclr_state_dict(self):
        model = ResNetSimCLR(base_model='resnet18', out_dim=128)
        modelsd = torch.load(self.config['feature']['path']['simclr_model_path'])
        model.load_state_dict(modelsd['state_dict'])
        model.eval()