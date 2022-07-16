# -*- coding: utf-8 -*-
# import os
# os.system('pip install pycocotools')
from collections import OrderedDict
import numpy as np
import mmcv
import cv2
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.core import encode_mask_results

import torch
import torch.nn.functional as F
from model_service.pytorch_model_service import PTServingBaseService

import time
from metric.metrics_manager import MetricsManager
from PIL import Image
import log
logger = log.getLogger(__name__)
import json


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        print(self.model_path)
        self.cfg = mmcv.Config.fromfile(self.model_path.replace('model_best.pth', '/configs/htc/htc_wrc.py'))
        print("model initing")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = init_detector(self.cfg, self.model_path, device="cpu")
        print("model already")


    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                imgarr = np.array(img)
                temp_path = self.model_path.replace('model_best.pth', 'test.jpg')
                cv2.imwrite(temp_path, imgarr)
                preprocessed_data[k] = temp_path
        return preprocessed_data

    def _inference(self, data):
        img_path = data["input_img"]
        res = inference_detector(self.model, img_path)
        bbox_results, mask_results = res
        mask_results = encode_mask_results(mask_results)
        outres = []
        for clsid, boxarrs in enumerate(bbox_results):
            for boxid, box in enumerate(boxarrs):
                res_dict = {}
                res_dict["image_id"] = 0
                box = box.tolist()
                res_dict["bbox"] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
                res_dict["score"] = box[4]
                res_dict["category_id"] = clsid
                res_dict["segmentation"] = mask_results[clsid][boxid]
                res_dict["segmentation"]["counts"] = res_dict["segmentation"]["counts"].decode()
                outres.append(res_dict)
        result = {"result": outres}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
        data = self._inference(data)
        print(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)
        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data