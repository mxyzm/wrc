import warnings

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import os
import cv2
from tqdm import tqdm
from skimage import io
import rasterio
from io import BytesIO
import base64
from PIL import Image
from math import ceil, floor
Image.MAX_IMAGE_PIXELS = None


def gettif(tif_path):
    try:
        data = cv2.imread(tif_path, -1)
        _shape = data.shape
    except:
        data = io.imread(tif_path)
    try:
        _shape = data.shape
    except:
        print("use rasterio")
        with rasterio.open(tif_path) as ds:
            band_nums = ds.count
            band1 = ds.read(1)
            if band_nums == 1:
                data = band1
            else:
                shape0, shape1 = band1.shape
                data = np.zeros((shape0, shape1, band_nums), dtype=band1.dtype).astype(band1.dtype)
                for i in range(band_nums):
                    data[:,:,band_nums-i-1] = ds.read(i+1)
    return data


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                        'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    def __call__(self, results):
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                    'future releases. You may use `LoadImageFromWebcam` '
                    'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, imgs):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results


def infer(cfg_path, model_path, test_dir, save_dir):
    cfg = mmcv.Config.fromfile(cfg_path)
    model = init_detector(cfg, model_path)
    print("model ready")
    file_name_list = os.listdir(test_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    for i, file_name in enumerate(file_name_list):
        file_path = os.path.join(test_dir, file_name)
        data = gettif(file_path)
        data = cv2.copyMakeBorder(data, 128, 128, 128, 128,cv2.BORDER_REFLECT_101)
        """这里进行切分预测"""
        """4_6加入膨胀预测"""
        x, y, c = data.shape
        label = np.zeros((x, y)).astype(np.uint8)
        target_l = 512
        step = 256
        x_num = ((x-target_l)//step + 2) if (x-target_l)%step else ((x-target_l)//step + 1)
        y_num = ((y-target_l)//step + 2) if (y-target_l)%step else ((y-target_l)//step + 1)
        for i in tqdm(range(x_num)):
            for j in range(y_num):
                x_s, x_e = i*step, i*step+target_l
                if x_e > x:
                    x_s, x_e = x-target_l, x
                y_s, y_e = j*step, j*step+target_l
                if y_e > y:
                    y_s, y_e = y-target_l, y
                img = data[x_s:x_e, y_s:y_e, :]
                """这里加入入网逻辑，全为0的不入网"""
                if (img==0).all():
                    continue
                res = inference_detector(model, img)[0]
                out_l = np.zeros((512, 512)).astype(np.uint8)
                for bnum in range(len(res)):
                    xmin, ymin, xmax, ymax, sco = res[bnum, :]
                    xmin, ymin, xmax, ymax = floor(xmin), floor(ymin), ceil(xmax), ceil(ymax)
                    out_l[ymin:ymax, xmin:xmax] = 1
                label[x_s+128:x_e, y_s+128:y_e] = out_l[128:, 128:].astype(np.uint8)
        label = label[128:-128, 128:-128]
        print(label.shape, label.dtype)
        save_path = os.path.join(save_dir, file_name)#, file_name.replace("tif", "png"))
        cv2.imwrite(save_path, label)


def tuban_eval(img_dir, gt_dir, save_dir):
    txt_path = os.path.join(save_dir, "eval.txt")
    f = open(txt_path, "w")
    name_list = os.listdir(save_dir)
    tp, fp, fn = 0, 0, 0
    for name in name_list:
        if not name.endswith("tif"): continue
        _tp, _fp, _fn = 0, 0, 0

        infer_ar = gettif(os.path.join(save_dir, name))
        gt_ar = gettif(os.path.join(gt_dir, name))
        img_ar = gettif(os.path.join(img_dir, name))
        gt_ar[(img_ar[:,:,0]==0)&(img_ar[:,:,1]==0)&(img_ar[:,:,2]==0)] = 0

        gt_retval, gt_labels, gt_stats, gt_centroids = cv2.connectedComponentsWithStats(gt_ar)
        in_retval, in_labels, in_stats, in_centroids = cv2.connectedComponentsWithStats(infer_ar)

        print("gt nums: ", gt_retval, "infer nums: ", in_retval)
        #遍历gt，如果infer中有就算一次tp，没有就算一次fn
        for i in tqdm(range(1, gt_retval)):
            if in_labels[gt_labels==i].any():
                _tp += 1
            else:
                _fn += 1
        _fp += (in_retval - _tp)
        print(_tp, _fn)
        f.write(name  + ": "+ str(_tp) +" "+ str(_fn) + " "+ str(_fp) + "\n")
        tp += _tp
        fp += _fp
        fn += _fn

    f.write("\n")
    f.write("tp: " + str(tp) +  "\n")
    f.write("fp: " + str(fp) +  "\n")
    f.write("fn: " + str(fn) +  "\n")
    f.write("recall: " + str(tp/(tp+fn)) +  "\n")
    f.write("pre: " + str(tp/(tp+fp)) +  "\n")


def ems(seg_dir, det_dir, em_save_dir, em_show_dir):
    if not os.path.exists(em_save_dir): os.makedirs(em_save_dir)
    if not os.path.exists(em_show_dir): os.makedirs(em_show_dir)
    name_list = os.listdir(seg_dir)
    for name in name_list:
        if not name.endswith("tif"): continue
        seg_label = gettif(os.path.join(seg_dir, name))
        det_label = gettif(os.path.join(det_dir, name))
        em_show_label = seg_label + det_label * 2
        em_label = np.zeros_like(seg_label)
        em_label[em_show_label >= 1] = 1
        cv2.imwrite(os.path.join(em_save_dir, name), em_label)
        cv2.imwrite(os.path.join(em_show_dir, name), em_show_label)


if __name__ == '__main__':
    cfg_path = "./configs/faster_rcnn/fsshuibao.py"
    model_path = "/home/ma-user/work/ckpt/shuibao_det/fasterrcnn_0503/latest.pth"
    test_dir = "/home/ma-user/work/data/shuibao/test/image/"
    save_dir = "/home/ma-user/work/data/shuibao/test/det/infer0504/"
    gt_dir = "/home/ma-user/work/data/shuibao/test/label/"


    # infer(cfg_path, model_path, test_dir, save_dir)
    # tuban_eval(gt_dir, save_dir)
    seg_dir = "/home/ma-user/work/data/shuibao/test/seg/infer0504/"
    det_dir = "/home/ma-user/work/data/shuibao/test/det/infer0504/"
    em_save_dir = "/home/ma-user/work/data/shuibao/test/ems/infer0504/"
    em_show_dir = "/home/ma-user/work/data/shuibao/test/ems_show/infer0504/"
    # ems(seg_dir, det_dir, em_save_dir, em_show_dir)
    tuban_eval(test_dir, gt_dir, em_save_dir)