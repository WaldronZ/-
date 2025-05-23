# 混合式主干-UPSNet（全景分割模型）




# Disclaimer

This repository is tested under Python 3.6, PyTorch 0.4.1. And model training is done with 16 GPUs by using [horovod](https://github.com/horovod/horovod). It should also work under Python 2.7 / PyTorch 1.0 and with 4 GPUs.



# Main Results

COCO 2017 (trained on train-2017 set)

|                   | test split | PQ   | SQ   | RQ   | PQ<sup>Th</sup> | PQ<sup>St</sup> |
| ----------------- | ---------- | ---- | ---- | ---- | --------------- | --------------- |
| 混合式主干-UPSNet | val        | 37.6 | 75.9 | 48.0 | 49.8            | 35.4            |

# Requirements: Software

We recommend using Anaconda3 as it already includes many common packages.


# Requirements: Hardware

We recommend using 4~16 GPUs with at least 11 GB memory to train our model.

# Installation

1.Clone this repo to `$UPSNet_ROOT`

2.Run `init.sh` to build essential C++/CUDA modules and download pretrained model.

For COCO:

Assuming you already downloaded COCO dataset at `$COCO_ROOT` and have `annotations` and `images` folders under it, please create a soft link by `ln -s $COCO_ROOT data/coco` under `UPSNet_ROOT`, and run `init_coco.sh` to prepare COCO dataset for UPSNet.

3.`conda create -n upsnet python=3.6` 

4.`conda install ./cudatoolkit-9.2-0.tar.bz2`

6.`pip install -r requirements.txt` 

`pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html`

7.

I have provide serveral config files (16/4 GPUs for Cityscapes/COCO dataset) under upsnet/experiments folder.

8.upsnet/models中新建efficientnet.py 和efficientnet_upsnet.py



9.upsnet/dataset/base_dataset.py中_pq_compute_single_core函数修改如下：

```python
    @staticmethod
    def _pq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories):
        OFFSET = 256 * 256 * 256
        VOID = 0
        pq_stat = PQStat()
        for idx, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(zip(gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set)):
            # if idx % 100 == 0:
            #     logger.info('Compute pq -> Core: {}, {} from {} images processed'.format(proc_id, idx, len(gt_jsons_set)))
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256
            if pan_gt.shape != pan_pred.shape:
                continue
            gt_segms = {el['id']: el for el in gt_json['segments_info']}
            pred_segms = {el['id']: el for el in pred_json['segments_info']}

            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in categories:
                    raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

            # confusion matrix calculation
            pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_matched = set()
            pred_matched = set()
            tp = 0
            fp = 0
            fn = 0

            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                if gt_label not in gt_segms:
                    continue
                if pred_label not in pred_segms:
                    continue
                if gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                    continue

                union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                    (VOID, pred_label), 0)
                iou = intersection / union
                if iou > 0.5:
                    pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                    pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    tp += 1

            # count false positives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                pq_stat[gt_info['category_id']].fn += 1
                fn += 1

            # count false positives
            for pred_label, pred_info in pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                pq_stat[pred_info['category_id']].fp += 1
                fp += 1
        # logger.info('Compute pq -> Core: {}, all {} images processed'.format(proc_id, len(gt_jsons_set)))
        return pq_stat
```



Training:

`python upsnet/upsnet_end2end_train.py --cfg upsnet/experiments/upsnet_efficientnetb0_coco_4gpu.yaml`

Test:

`python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_efficientnetb0_coco_4gpu.yaml`





# Model Weights

For COCO:

```shell
python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_efficientnetb0_coco_4gpu.yaml --weight_path model/pretrained_model/efficientnet230000.pth
```

```shell
python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet101_dcn_coco_3x_16gpu.yaml --weight_path model/pretrained_model/efficientnet230000.pth
```







