model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=28,
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=3,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(),
    test_cfg=dict(average_clips='prob'))
dataset_type = 'PoseDataset'
ann_file_train = '/home/pepeotalk/mmaction2/data/user_and_aihub_crop/result-train-turn-shake-scratch.pkl'
ann_file_test = '/home/pepeotalk/mmaction2/data/user_and_aihub_crop/result-test-shake-scratch.pkl'
left_kp = [0, 1, 4, 10, 11, 12, 13, 19, 20, 21]
right_kp = [2, 3, 5, 13, 14, 15, 22, 23, 24]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=10, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='PoseDataset',
        ann_file=
        '/home/pepeotalk/mmaction2/data/user_and_aihub_crop/result-train-turn-shake-scratch.pkl',
        data_prefix='',
        pipeline=[
            dict(type='UniformSampleFrames', clip_len=10),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(-1, 64)),
            dict(type='Resize', scale=(56, 56), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    test=dict(
        type='PoseDataset',
        ann_file=
        '/home/pepeotalk/mmaction2/data/user_and_aihub_crop/result-test-shake-scratch.pkl',
        data_prefix='',
        pipeline=[
            dict(
                type='UniformSampleFrames',
                clip_len=10,
                num_clips=1,
                test_mode=True),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(-1, 64)),
            dict(type='CenterCrop', crop_size=64),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=True,
                with_limb=False,
                double=True),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 500
checkpoint_config = dict(interval=10)
workflow = [('train', 10)]
evaluation = dict(
    interval=10,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/posec3d/slowonly_r50_u48_500e_aihub+user_crop_keypoint'
load_from = None
resume_from = None
find_unused_parameters = False
gpu_ids = range(0, 2)
omnisource = False
module_hooks = []
