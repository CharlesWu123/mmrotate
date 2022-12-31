# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/12/28 19:04
@File: dataset_wrappers.py
@Desc: 
"""
import collections
import copy

from .builder import ROTATED_DATASETS, ROTATED_PIPELINES
from mmdet.datasets import MultiImageMixDataset
from mmcv.utils import build_from_cfg, print_log


@ROTATED_DATASETS.register_module()
class MyMultiImageMixDataset(MultiImageMixDataset):
    def __init__(self,
                 dataset,
                 pipeline,
                 dynamic_scale=None,
                 skip_type_keys=None,
                 max_refetch=15):
        if dynamic_scale is not None:
            raise RuntimeError(
                'dynamic_scale is deprecated. Please use Resize pipeline '
                'to achieve similar functions')
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = build_from_cfg(transform, ROTATED_PIPELINES)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.indices = []
        self.dataidx = []
        if not isinstance(dataset, (list, tuple)):
            dataset = [dataset]
        for i, d in enumerate(dataset):
            self.indices += [i] * len(d)
            self.dataidx += list(range(len(d)))
        self.dataset = dataset
        self.CLASSES = dataset[0].CLASSES
        self.PALETTE = getattr(dataset[0], 'PALETTE', None)
        if hasattr(self.dataset[0], 'flag'):
            self.flag = dataset[0].flag
        self.num_samples = len(self.indices)
        self.max_refetch = max_refetch

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        dataset = self.dataset[self.indices[idx]]
        if len(self.dataset) == 2:
            if self.indices[idx] == 0:
                ratio = [1, 2]
            else:
                ratio = [2, 1]
        else:
            ratio = [3]
        # 增加功能，当 mosaic 不是第一个transform时，每次 mosaic 都要执行之前的 transform
        results = copy.deepcopy(dataset[self.dataidx[idx]])
        pre_transform = []
        for (transform, transform_type) in zip(self.pipeline, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                for i in range(self.max_refetch):
                    # Make sure the results passed the loading pipeline
                    # of the original dataset is not None.
                    indexes = transform.get_indexes(self.dataset, ratio)
                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]
                    # mix_results = [
                    #     copy.deepcopy(self.dataset[index]) for index in indexes
                    # ]
                    mix_results = []
                    for index in indexes:
                        data = copy.deepcopy(self.dataset[self.indices[index]][self.dataidx[index]])
                        for t in pre_transform:
                            data = t(data)
                        mix_results.append(data)
                    if None not in mix_results:
                        results['mix_results'] = mix_results
                        break
                else:
                    raise RuntimeError(
                        'The loading pipeline of the original dataset'
                        ' always return None. Please check the correctness '
                        'of the dataset and its pipeline.')
                pre_transform = []
            for i in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                updated_results = transform(copy.deepcopy(results))
                if updated_results is not None:
                    results = updated_results
                    break
            else:
                raise RuntimeError(
                    'The training pipeline of the dataset wrapper'
                    ' always return None.Please check the correctness '
                    'of the dataset and its pipeline.')

            if 'mix_results' in results:
                results.pop('mix_results')
            pre_transform.append(transform)
        return results
