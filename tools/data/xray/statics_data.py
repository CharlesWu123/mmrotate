# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/12/27 10:36
@File: statics_data.py
@Desc: 
"""
import json
import os
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

CLASSES = (
    'knife', 'pressure', 'umbrella', 'lighter', 'OCbottle',
    'glassbottle', 'battery', 'metalbottle', 'electronicequipment'
)


def read_anno(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cls_name = []
    anno = []
    for line in lines:
        line_s = line.strip().split()
        cls_name.append(line_s[2])
        anno.append({
            'name': line_s[2],
            'poly': [
                [int(line_s[7]), int(line_s[8])],
                [int(line_s[9]), int(line_s[10])],
                [int(line_s[11]), int(line_s[12])],
                [int(line_s[13]), int(line_s[14])]
            ]
        })
    return cls_name, anno


def class_distribute(anno_dir):
    all_classes = []
    anno_file_list = os.listdir(anno_dir)
    for anno_file in anno_file_list:
        all_classes.extend(read_anno(os.path.join(anno_dir, anno_file))[0])
    counter = Counter(all_classes)
    class_dist = dict(counter.most_common())
    df = pd.DataFrame(class_dist.values(), index=class_dist.keys(), columns=['数量'])
    total = df['数量'].sum()
    df['占比'] = df['数量'].map(lambda x: format(x/total, '.2%'))
    print(df)
    # df.plot()
    # plt.show()


def edge_ratio(anno_dir):
    anno_file_list = os.listdir(anno_dir)
    ratio_list = []
    ratio_map = {}
    name_map = {}
    for anno_file in anno_file_list:
        _, anno = read_anno(os.path.join(anno_dir, anno_file))
        for line in anno:
            name = line['name']
            points = np.array(line['poly'])
            img_crop_width = int(
                max(np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3])))
            img_crop_height = int(
                max(np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2])))
            if img_crop_height == 0 or img_crop_width == 0: continue
            if img_crop_height > img_crop_width:
                ratio = img_crop_height / img_crop_width
            else:
                ratio = img_crop_width / img_crop_height
            ratio_list.append(int(ratio))
            if int(ratio) not in ratio_map:
                ratio_map[int(ratio)] = []
            ratio_map[int(ratio)].append(name)
            if name not in name_map:
                name_map[name] = []
            name_map[name].append(int(ratio))
    print('=' * 10 + 'ratio -> name' + '=' * 10)
    for n, name_list in ratio_map.items():
        # ratio_map[n] = Counter(name_list).most_common()
        print(n, ':', Counter(name_list).most_common())
    print('=' * 10 + 'name -> ratio' + '=' * 10)
    for name, n_list in name_map.items():
        # name_map[name] = Counter(n_list).most_common()
        print(name, ':', Counter(n_list).most_common())
    print('=' * 10 + 'ratio counter' + '=' * 10)
    counter = Counter(ratio_list)
    class_dist = dict(counter.most_common())
    df = pd.DataFrame(class_dist.values(), index=class_dist.keys(), columns=['数量'])
    total = df['数量'].sum()
    df['占比'] = df['数量'].map(lambda x: format(x / total, '.2%'))
    print(df)


if __name__ == '__main__':
    anno_dir = '/data/wuzhichao/homework/rotate_data/datasets_gen/datasets/annotations_patched'
    # class_distribute(anno_dir)
    edge_ratio(anno_dir)
"""
train
knife                2159  24.45%
electronicequipment  1412  15.99%
lighter              1378  15.60%
battery               984  11.14%
OCbottle              899  10.18%
metalbottle           737   8.34%
pressure              490   5.55%
glassbottle           432   4.89%
umbrella              341   3.86%

val

gen
knife                2395  19.03%
lighter              1889  15.01%
electronicequipment  1579  12.55%
battery              1490  11.84%
OCbottle             1365  10.85%
metalbottle          1278  10.15%
glassbottle          1204   9.57%
pressure              787   6.25%
umbrella              598   4.75%
"""