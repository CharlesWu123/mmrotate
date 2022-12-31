# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/12/28 21:28
@File: gen_anno_check.py
@Desc: 
"""
import os

anno_dir = '/data/wuzhichao/homework/rotate_data/datasets_hw/train/annotations_patched'
anno_file = os.listdir(anno_dir)
for file in anno_file:
    file_path = os.path.join(anno_dir, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    data = data.replace('ocbottle', 'OCbottle')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(data)
