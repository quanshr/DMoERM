import os
from base_config import *

cat_list = ['角色扮演', '闲聊', '主观知识问答', '客观知识问答', '文本创作']

out_dir = os.path.join(out_dir, 'router')
os.makedirs(out_dir, exist_ok=True)

val_rate = 0.2
eval_samples = 500
max_no_adding_times = 200
steps_per_eval = 100

lr = 3e-8
