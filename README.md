# Sample For Car License Recognization

## Purpose
This sample is to show how to use graded models for detection and classification with DeepStream SDK version not less than 5.0.1. The models in this sample are all TLT3.0 models.

`PGIE(car detection) -> SGIE(car license plate detection) -> SGIE(car license plate recognization)`

## Preparation For The Sample
Before using this car license recognition sample, the DeepStream SDK should be installed correctly. The installation instruction and user manual can be found in [DeepStream SDK manual](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html), after installation, it is better to run deepstream-test1-app to test whether the installation is successful.
The new tlt-converter tool is also needed for the sample model engine build. Download the latest tlt-converter for your appropriate Hardware/CUDA/cuDNN version from the [TLT getting started](https://developer.nvidia.com/tlt-getting-started) page

## Introduction of The Sample
This sample uses three models to build the unction of recognizing characters on the car plates. 
The car detection model:
For US car detection, refer to the tlt pretrained trafficcamnet model. The model can be downloaded with the instruction of /opt/nvidia/deepstream/deepstream-5.0/samples/configs/tlt_pretrained_models/README
For Chinese car detection, the new tlt models are available. 

The car license plate detection model is LPD model.

The car license plate recognization model is LPR models:
With DeepStreamSDK 5.0.1, the gst-nvinfer can not generate trt engine with the new TLT models. The engine files should be generated with the new tlt-converter tool.
The sample command line for the onnx model is:

`./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 ./us_lprnet_baseline18_deployable.etlt -t fp16 -e lpr_us_onnx_b16.engine`

The application uses the above three models to recognize the car plate characters in video or pictures.

![LPR/LPD application](lpr.png)

Test recognition result can be shown on the video.
![result](result.png)

The following test is done with 1080p(1920x1080) resolution videos on T4 with the sample LPR application. The table below shows the end-to-end performance of processing the entire video analytic pipeline with 3 DNN models - starting from ingesting video data to rendering the metadata on the frames. The data is collected on different devices. 

| Device    | Number of streams | Batch Size | Total FPS |
|-----------| ----------------- | -----------|-----------|
|Jetson Nano|     1             |     1      | 9.2       |
|Jetson NX  |     3             |     3      | 80.31     |
|Jetson Xavier |  5             |     5      | 146.43    |
|T4         |     14            |     14     | 447.15    |

More details for TLT LPD/LPR models and training, please refer to [APLR blog](https://docs.google.com/document/d/1tMH0ku284AqqcVdioS1XazyT0-uGNNpg4-r64JaIBZA/edit#).

## Build And Run
* Download the car detection, plate detection and plate recognition models to the board.

  ```mkdir -p /opt/nvidia/deepstream/deepstream-5.0/samples/models/tlt_pretrained_models/trafficcamnet```

  ```cd /opt/nvidia/deepstream/deepstream-5.0/samples/models/tlt_pretrained_models/trafficcamnet```

  ```wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_trafficcamnet/versions/pruned_v1.0/files/trafficnet_int8.txt```

  ```wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_trafficcamnet/versions/pruned_v1.0/files/resnet18_trafficcamnet_pruned.etlt```
   
  ```mkdir -p /opt/nvidia/deepstream/deepstream-5.0/samples/models/LP/LPD```

  ```cd /opt/nvidia/deepstream/deepstream-5.0/samples/models/LP/LPD```

  ```wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lpdnet/versions/pruned_v1.0/files/usa_pruned.etlt```

  ```wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lpdnet/versions/pruned_v1.0/files/usa_lpd_cal.bin```
  
  ```wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lpdnet/versions/pruned_v1.0/files/usa_lpd_label.txt```
   
  ```mkdir -p /opt/nvidia/deepstream/deepstream-5.0/samples/models/LP/LPR```

  ```cd /opt/nvidia/deepstream/deepstream-5.0/samples/models/LP/LPR```

  ```wget https://api.ngc.nvidia.com/v2/models/nvidia/tlt_lprnet/versions/deployable_v1.0/files/us_lprnet_baseline18_deployable.etlt```
  
  ```echo > labels_us.txt```
  
  With DeepStreamSDK 5.0.1, the gst-nvinfer can not generate trt engine with the new TLT models. The engine files should be generated with the new tlt-converter tool.
The sample command line for the onnx model is:

  ```./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 ./us_lprnet_baseline18_deployable.etlt -t fp16 -e lpr_us_onnx_b16.engine```
   
* Download the sample code to the directory of /opt/nvidia/deepstream/deepstream-5.0/sources/apps
   Build nvinfer_custom_lpr_parser library.
   
  ```cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/ds-lpr-sample/nvinfer_custom_lpr_parser```

  ```sudo make```

  ```sudo cp libnvdsinfer_custom_impl_lpr.so ../../../../lib/```
 
* Build lpr-test-sample

  ```cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/ds-lpr-sample/lpr-test-sample```

  ```sudo make```
* Prepare dictionary file for US car plate strings

  ```cd /opt/nvidia/deepstream/deepstream-5.0/sources/apps/ds-lpr-sample/lpr-test-sample```

  ```cp dict_us.txt dict.txt```

* Run the application with the following command line:

  ```./lpr-test-app [language mode:1-us 2-chinese] [sink mode:1-output as 264 stream file 2-no output 3-display on screen] [0:ROI disable|0:ROI enable] [input mp4 file path and name] [input mp4 file path and name] ... [input mp4 file path and name] [output 264 file path and name]```
