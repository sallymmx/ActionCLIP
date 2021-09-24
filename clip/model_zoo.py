# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
def get_model_path(ckpt):
    if os.path.isfile(ckpt):
        return ckpt
    else:
        print('not found pretrained model in {}'.format(ckpt))
        raise FileNotFoundError
