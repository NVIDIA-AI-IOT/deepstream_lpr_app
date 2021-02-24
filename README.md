# Sample For Car License Recognization
 - [Description](#description)
 - [Performance](#performance)
 - [Prerequisition](#prerequisition)
 - [Download](#download)
 - [Build and Run](#build-and-run)
 - [Notice](#notice)

---

## Description
This sample is to show how to use graded models for detection and classification with DeepStream SDK version not less than 5.0.1. The models in this sample are all TLT3.0 models.

`PGIE(car detection) -> SGIE(car license plate detection) -> SGIE(car license plate recognization)`

![LPR/LPD application](lpr.png)

This pipeline is based on three TLT models below

* Car detection model https://ngc.nvidia.com/catalog/models/nvidia:tlt_trafficcamnet
* LPD (car license plate detection) model https://ngc.nvidia.com/catalog/models/nvidia:tlt_lpdnet
* LPR (car license plate recognization/text extraction) model https://ngc.nvidia.com/catalog/models/nvidia:tlt_lprnet

More details for TLT3.0 LPD and LPR models and TLT training, please refer to [TLT document](https://docs.nvidia.com/metropolis/TLT/tlt-getting-started-guide/).

## Performance
Below table shows the end-to-end performance of processing 1080p videos with this sample application.

| Device    | Number of streams | Batch Size | Total FPS |
|-----------| ----------------- | -----------|-----------|
|Jetson Nano|     1             |     1      | 9.2       |
|Jetson NX  |     3             |     3      | 80.31     |
|Jetson Xavier |  5             |     5      | 146.43    |
|T4         |     14            |     14     | 447.15    |

## Prerequisition

* [DeepStream SDK 5.0.1](https://developer.nvidia.com/deepstream-getting-started)

  Make sure deepstream-test1 sample can run successful to verify your DeepStream installation
  
* [tlt-converter](https://developer.nvidia.com/tlt-getting-started)

  Download x86 or Jetson tlt-converter which is compatible to your platform from the following links.

| Platform   |  Compute                       |        Link                                              |
|------------|--------------------------------|----------------------------------------------------------|
|x86 + GPU   |CUDA 10.2/cuDNN 8.0/TensorRT 7.1|[link](https://developer.nvidia.com/cuda102-cudnn80-trt71)|
|x86 + GPU   |CUDA 10.2/cuDNN 8.0/TensorRT 7.2|[link](https://developer.nvidia.com/cuda102-cudnn80-trt72)|
|x86 + GPU   |CUDA 11.0/cuDNN 8.0/TensorRT 7.1|[link](https://developer.nvidia.com/cuda110-cudnn80-trt71)|
|x86 + GPU   |CUDA 11.0/cuDNN 8.0/TensorRT 7.2|[link](https://developer.nvidia.com/cuda110-cudnn80-trt72)|
|Jetson      |JetPack 4.4                     |[link](https://developer.nvidia.com/cuda102-trt71-jp44)   |
|Jetson      |JetPack 4.5                     |[link](https://developer.nvidia.com/cuda102-trt71-jp45)   |

## Download

1. Download Project with SSH or HTTPS
```
    // SSH
    git clone git@github.com:NVIDIA-AI-IOT/deepstream_lpr_app.git
    // or HTTPS
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app.git
```
2. Prepare Models and TensorRT engine

```
    cd deepstream_lpr_app/
```
For US car plate recognition
```
    ./download_us.sh
    // DS5.0.1 gst-nvinfer cannot generate TRT engine for LPR model, so generate it with tlt-converter
    ./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
           models/LP/LPR/us_lprnet_baseline18_deployable.etlt -t fp16 -e models/LP/LPR/lpr_us_onnx_b16.engine
```
For Chinese car plate recognition
```
    ./download_ch.sh
    // DS5.0.1 gst-nvinfer cannot generate TRT engine for LPR model, so generate it with tlt-converter
    ./tlt-converter -k nvidia_tlt -p image_input,1x3x48x96,4x3x48x96,16x3x48x96 \
           models/LP/LPR/ch_lprnet_baseline18_deployable.etlt -t fp16 -e models/LP/LPR/lpr_ch_onnx_b16.engine
```

## Build and Run
```
    make
    cd deepstream-lpr-app
```
For US car plate recognition
```
    cp dict_us.txt dict.txt
```
For Chinese car plate recognition
```
    cp dict_ch.txt dict.txt
```
Start to run the application
```
    ./deepstream-lpr-app <1:US car plate model|2: Chinese car plate model> \
         <1: output as h264 file| 2:fakesink 3:display output> <0:ROI disable|1:ROI enable> \
         <input mp4 file name> ... <input mp4 file name> <output file name>
```
A sample of US car plate recognition:

`./deepstream-lpr-app 1 2 0 us_car_test2.mp4 us_car_test2.mp4 output.264`

A sample of Chinese car plate recognition:

`./deepstream-lpr-app 2 2 0 ch_car_test.mp4 ch_car_test.mp4 output.264`

## Notice
1. This sample application only support mp4 files which contain H264 videos as input files.
2. For Chinese plate recognition, please make sure the OS supports Chinese language.
3. The second argument of deepstream-lpr-app should be 2(fakesink) for performance test.
4. The trafficcamnet and LPD models are all INT8 models, the LPR model is FP16 model.
