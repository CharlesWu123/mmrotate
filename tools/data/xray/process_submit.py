# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/12/26 22:01
@File: process_submit.py
@Desc: 
"""
from natsort import natsorted

file_path = '/data/wuzhichao/homework/mmrotate/submission_dir/20221227_104824/results_src.txt'
save_path = '/data/wuzhichao/homework/mmrotate/submission_dir/20221227_104824/results.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    split_line = line.split()
    if float(split_line[2]) < 0.1: continue
    new_lines.append(split_line)
new_lines = natsorted(new_lines, key=lambda x: (x[0], x[1]))
new_file = open(save_path, 'w', encoding='utf-8')
for line in new_lines:
    new_file.writelines(' '.join(line) + '\n')
new_file.close()
