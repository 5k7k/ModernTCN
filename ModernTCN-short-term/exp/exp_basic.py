import os
import torch

from models import ModernTCN, PatchTST, TimeMixer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'ModernTCN':ModernTCN,
            'PatchTST':PatchTST,
            'TimeMixer':TimeMixer,

        }
        if self.args.model not in self.model_dict:
            raise ValueError('Unsupported model: {}. Available models: {}'.format(
                self.args.model, list(self.model_dict.keys())))
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
