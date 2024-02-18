# Real-Time Anomaly Segmentation for Road Scenes
This repository contains the code of the __Real-Time Anomaly Segmentation for Road Scenes__ project of the __Advanced Machine Learning__ course 23/24 - Politecnico di Torino

### Sample Results

#### First Example

* Original Image <br/>
<img src="eval/saved_anomalies/tractor.png" alt="Tractor" style="height:128px;width:256px;"/>

* Ground Truth Anomaly <br/>
<img src="eval/saved_anomalies/tractor_label.png" alt="Tractor Ground Truth Anomaly" style="height:128px;width:256px;"/>

* Anomaly Scores <br/>
<img src="eval/saved_anomalies/tractor_anomaly_scores.png" alt="Tractor Anomaly Scores" style="height:128px;width:256px;"/>

#### Second Example

* Original Image <br/>
<img src="eval/saved_anomalies/phone_box.png" alt="Phone Box" style="height:128px;width:256px;"/>

* Ground Truth Anomaly <br/>
<img src="eval/saved_anomalies/phone_box_label.png" alt="Phone Box Truth Anomaly" style="height:128px;width:256px;"/>

* Anomaly Scores <br/>
<img src="eval/saved_anomalies/phone_box_anomaly_scores.png" alt="Phone Box Anomaly Scores" style="height:128px;width:256px;"/>

## Packages
For instructions, please refer to the __README__ in each folder:

* [train](train) contains tools for training the networks for semantic segmentation.
* [eval](eval) contains tools for evaluating/visualizing the networks' output and performing anomaly segmentation.
* [imagenet](imagenet) contains scripts and model for pretraining ERFNet's encoder in Imagenet.
* [trained_models](trained_models) contains some trained models used in the papers (almost all the models are available in the [Releases section](https://github.com/FarInHeight/Real-Time-Anomaly-Segmentation-for-Road-Scenes/releases/tag/v3.0.0)). 

## Datasets

* [**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. **Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds**
* **For testing the anomaly segmentation models**: All testing images are provided [here](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view).

## Networks
The repo provides the following pre-trained networks that can be used to perform anomaly segmentation:
* __Erfnet__ trained on 19 classes of the Cityscapes dataset using a __Cross-Entropy loss__, __Logit Norm + Cross Entropy__, __Logit Norm + Focal Loss__, __IsoMax+ + Cross Entropy__ and __IsoMax+ + Focal Loss__
* __BiSeNetV1__ trained on 20 classes (19 + void class) of the Cityscapes dataset
* __Enet__ trained on 20 classes (19 + void class) of the Cityscapes dataset


## Anomaly Inference
To run the anomaly inferences method is possible to use the following command
* Anomaly Inference Command: ```python evalAnomaly.py --input='/content/validation_dataset/RoadAnomaly21/images/*.png'```. Change the dataset path ```'/content/validation_dataset/RoadAnomaly21/images/*.png'``` accordingly.

## Notebook
The `AML_Project.ipynb` can be opened on Colab to run all the evaluation commands.

## Authors

- [Davide Sferrazza s326619](https://github.com/FarInHeight/)
- [Davide Vitabile s330509](https://github.com/Vitabile/)
- [Yonghu Liu s313442](https://github.com/Liu-Yonghu)

## License
[MIT License](LICENSE)