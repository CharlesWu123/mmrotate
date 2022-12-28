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

import pandas as pd
import matplotlib.pyplot as plt


CLASSES = (
    'knife', 'pressure', 'umbrella', 'lighter', 'OCbottle',
    'glassbottle', 'battery', 'metalbottle', 'electronicequipment'
)


def read_anno(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cls_name = []
    for line in lines:
        line_s = line.strip().split()
        cls_name.append(line_s[2])
    return cls_name


def class_distribute(anno_dir):
    all_classes = []
    anno_file_list = os.listdir(anno_dir)
    for anno_file in anno_file_list:
        all_classes.extend(read_anno(os.path.join(anno_dir, anno_file)))
    counter = Counter(all_classes)
    class_dist = dict(counter.most_common())
    df = pd.DataFrame(class_dist.values(), index=class_dist.keys(), columns=['数量'])
    total = df['数量'].sum()
    df['占比'] = df['数量'].map(lambda x: format(x/total, '.2%'))
    # df.plot()
    # plt.show()


if __name__ == '__main__':
    anno_dir = '/data/wuzhichao/homework/rotate_data/datasets_hw/train/annotations'
    class_distribute(anno_dir)

"""
train
{
  "knife": 2159,
  "electronicequipment": 1412,
  "lighter": 1378,
  "battery": 984,
  "OCbottle": 899,
  "metalbottle": 737,
  "pressure": 490,
  "glassbottle": 432,
  "umbrella": 341
}
val
{
  "knife": 693,
  "lighter": 461,
  "electronicequipment": 444,
  "battery": 303,
  "OCbottle": 281,
  "metalbottle": 250,
  "pressure": 173,
  "glassbottle": 133,
  "umbrella": 110
}
"""