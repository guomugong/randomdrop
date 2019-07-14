# Random drop loss for tiny object segmentation: Application to lesion segmentation in fundus images
Please read our [paper]() for more details.


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
