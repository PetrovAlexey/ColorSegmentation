# Semantic Segmentation Metrics on Pytorch

python demo.py find_metrics C:\Users\Alexey\Documents\psa\VOCdevkit\VOC2012\SegmentationClass C:\Users\Alexey\Documents\psa\VOCdevkit\VOC2012\SegmentationClass C:\Users\Alexey\Documents\psa\VOCdevkit\VOC2012\ImageSets\Segmentation\temp.txt

## Metrics used:

* Pixel Accuracy
* mean Accuracy(of per-class pixel accuracy)
* mean IOU(of per-class Mean IOU)
* Frequency weighted IOU

### Calculate the metrics
Use

`python demo.py find_metrics predict_path gt_path id_file`