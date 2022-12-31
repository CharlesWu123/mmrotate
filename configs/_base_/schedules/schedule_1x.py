# evaluation
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',      # 优化策略
    warmup='linear',    # 初始的学习率增加的策略
    warmup_iters=500,   # 在初始的500次迭代中学习率逐渐增加
    warmup_ratio=1.0 / 3,       # 起始学习率
    step=[15, 18])      # 降低学习率的步数
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1, metric='bbox', save_best='auto')
