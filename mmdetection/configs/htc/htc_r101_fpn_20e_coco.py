_base_ = './htc_r50_fpn_1x_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)

load_from = "/home/ma-user/work/wrc_htc/pre/htc_r101_fpn_20e_coco_20200317-9b41b48f.pth"

