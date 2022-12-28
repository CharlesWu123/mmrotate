# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/12/26 19:57
@File: data_vis.py
@Desc: 
"""

import os
import cv2
import random
import numpy as np
from tqdm import tqdm

class_name = ['knife', 'pressure', 'umbrella', 'lighter', 'OCbottle', 'glassbottle', 'battery', 'metalbottle', 'electronicequipment']
colors = {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in class_name}


def read_anno(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    anno = []
    for line in lines:
        line_s = line.strip().split()
        anno.append({
            'name': line_s[2],
            'poly': [
                [int(line_s[7]), int(line_s[8])],
                [int(line_s[9]), int(line_s[10])],
                [int(line_s[11]), int(line_s[12])],
                [int(line_s[13]), int(line_s[14])]
            ]
        })
    return anno


def read_submit_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    anno = {}
    for line in lines:
        line_s = line.strip().split()
        img_name = line_s[0]
        if img_name not in anno:
            anno[img_name] = []
        anno[img_name].append({
            'name': line_s[1],
            'score': float(line_s[2]),
            'poly': [
                [int(line_s[3].split('.')[0]), int(line_s[4].split('.')[0])],
                [int(line_s[5].split('.')[0]), int(line_s[6].split('.')[0])],
                [int(line_s[7].split('.')[0]), int(line_s[8].split('.')[0])],
                [int(line_s[9].split('.')[0]), int(line_s[10].split('.')[0])],
            ]
        })
    return anno


def draw_img(img_path, anno, save_path, thickness=2):
    img = cv2.imread(img_path)
    # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)     # # windows下中文路径
    for obj in anno:
        name = obj['name']
        poly = obj['poly']
        cv2.polylines(img, np.array([poly]), True, colors[name], thickness=thickness)

        t_size = cv2.getTextSize(name, 0, fontScale=thickness / 3, thickness=thickness)[0]
        c2 = poly[0][0] + t_size[0], poly[0][1] - t_size[1] - 3
        cv2.rectangle(img, (poly[0][0], poly[0][1]), c2, colors[name], -1, cv2.LINE_AA)  # filled
        cv2.putText(img, name, (poly[0][0], poly[0][1] - 2), 0, thickness / 3, [225, 255, 255], thickness=thickness, lineType=cv2.LINE_AA)
    cv2.imwrite(save_path, img)
    # cv2.imencode('.jpg', img)[1].tofile(save_path)      # windows下中文路径


def vis_img(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    anno_dir = os.path.join(data_dir, 'annotations')
    image_dir = os.path.join(data_dir, 'images')
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        anno_path = os.path.join(anno_dir, os.path.splitext(image_name)[0] + '.txt')
        save_path = os.path.join(save_dir, image_name)
        draw_img(image_path, read_anno(anno_path), save_path)


def vis_img_test(res_file, image_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    anno_dict = read_submit_file(res_file)
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        save_path = os.path.join(save_dir, image_name)
        draw_img(image_path, anno_dict[image_name], save_path)


if __name__ == '__main__':
    # data_dir = r'D:\北航\学习资料\机器学习基础\大作业\大作业资料\datasets_hw\train'
    # save_dir = r'D:\北航\学习资料\机器学习基础\大作业\大作业资料\datasets_hw\train_vis'
    # vis_img(data_dir, save_dir)

    res_file = '/data/wuzhichao/homework/mmrotate/submission_dir/20221226_115609/results.txt'
    img_dir = '/data/wuzhichao/homework/rotate_data/datasets_hw/test/images'
    save_dir = '/data/wuzhichao/homework/rotate_data/datasets_hw/test/vis'
    vis_img_test(res_file, img_dir, save_dir)
