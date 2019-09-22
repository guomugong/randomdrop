# Random drop loss for tiny object segmentation: Application to lesion segmentation in fundus images
Please read our [paper](https://link.springer.com/chapter/10.1007%2F978-3-030-30508-6_18) for more details.

### Introduction:
Convolutional neural network (CNN), has achieved state-of-the-art performance in computer vision tasks. The segmentation of dense objects has been fully studies, but the research is insufficient on
tiny objects segmentation which is very common in medical images. For instance, the proportion of lesions or tumors can be as low as 0.1%,
which can easily lead to misclassification. In this paper, we propose a random drop loss function to improve the segmentation performance of
tiny lesions on medical image analysis task by dropping negative samples randomly according to their classification difficulty. In addition, we
designed three drop functions to map the classification difficulty to drop probability with the principle that easy negative samples are dropped
with high probabilities and hard samples are retained with high probabilities.  In this manner, not only can the sorting process existing in Top-k
BCE loss be avoided, but CNN can also learn better discriminative features, thereby reducing misclassification. We evaluated our method on
the task of segmentation of microaneurysms and hemorrhages in color fundus images. Experimental results show that our method outperforms
other methods in terms of segmentation performance and computational cost.

## Add the following statements to caffe.proto
```
optional SigmoidRandomDropCELossParameter randrop_loss_param = 208;

message SigmoidRandomDropCELossParameter {
  enum ScheduleType {
  LINEAR = 0;
  SQU    = 1;
  LOG    = 2;
  }
  optional ScheduleType schedule_type = 1 [default = LOG];
}
```

## Usage:
```
layer {
  name: "loss"
  type: "SigmoidRandomDropCELossParameter"
  bottom: "pred"
  bottom: "label"
  top: "loss"
  loss_weight: 1
  randrop_loss_param {
    schedule_type: LINEAR
  }
}
```
## License
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![Badge](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu/#/zh_CN)
