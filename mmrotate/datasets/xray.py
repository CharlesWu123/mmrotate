# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/12/26 13:41
@File: xray.py
@Desc: 
"""
import glob
import os
import os.path as osp
import time
import re
import mmcv

from collections import defaultdict
from functools import partial
import numpy as np
from mmcv.ops import nms_rotated
from natsort import natsorted

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS

from .dota import DOTADataset, _merge_func


@ROTATED_DATASETS.register_module()
class XRayDataset(DOTADataset):
    CLASSES = (
        'knife', 'pressure', 'umbrella', 'lighter', 'OCbottle',
        'glassbottle', 'battery', 'metalbottle', 'electronicequipment'
    )

    PALETTE = [
        (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0), (138, 43, 226),
        (255, 128, 0), (255, 0, 255), (0, 255, 255), (255, 193, 193)
    ]

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.jpg')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[7:], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[2]
                        difficulty = 0
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            # splitname = img_id.split('__')
            # oriname = splitname[0]
            # pattern1 = re.compile(r'__\d+___\d+')
            # x_y = re.findall(pattern1, img_id)
            # x_y_2 = re.findall(r'\d+', x_y[0])
            # x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                # ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                #     [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[img_id].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        os.makedirs(out_folder, exist_ok=True)
        save_file_path = os.path.join(out_folder, 'results.txt')
        file = open(save_file_path, 'w')
        lines = []
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for cls, dets in enumerate(dets_per_cls):
                if dets.size == 0: continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    if bbox[-1] < 0.1: continue
                    text_element = [str(img_id) + '.jpg', self.CLASSES[cls], str(bbox[-1])] + [f'{p:.2f}' for p in bbox[:-1]]
                    lines.append(text_element)
                    # file.writelines(' '.join(text_element) + '\n')
        lines = natsorted(lines, key=lambda x: (x[0], x[1]))
        for line in lines:
            file.writelines(' '.join(line) + '\n')
        file.close()
        return [save_file_path]
