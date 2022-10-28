# Sample For Car License Recognization
 - [Description](#description)
 - [Performance](#performance)
 - [Prerequisition](#prerequisition)
 - [Download](#download)
 - [Build and Run](#build-and-run)
 - [Notice](#notice)

---

## Description
This sample is to show how to use graded models for detection and classification with DeepStream SDK version not less than 5.0.1. The models in this sample are all TAO3.0 models.

`PGIE(car detection) -> SGIE(car license plate detection) -> SGIE(car license plate recognization)`

![LPR/LPD application](lpr.png)

This pipeline is based on three TAO models below

* Car detection model https://ngc.nvidia.com/catalog/models/nvidia:tao:trafficcamnet
* LPD (car license plate detection) model https://ngc.nvidia.com/catalog/models/nvidia:tao:lpdnet
* LPR (car license plate recognization/text extraction) model https://ngc.nvidia.com/catalog/models/nvidia:tao:lprnet

More details for TAO3.0 LPD and LPR models and TAO training, please refer to [TAO document](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html).

## Performance
Below table shows the end-to-end performance of processing 1080p videos with this sample application.

| Device    | Number of streams | Batch Size | Total FPS |
|-----------| ----------------- | -----------|-----------|
|Jetson Nano|     1             |     1      | 9.2       |
|Jetson NX  |     3             |     3      | 80.31     |
|Jetson Xavier |  5             |     5      | 146.43    |
|Jetson Orin|     5             |     5      | 341.65    |
|T4         |     14            |     14     | 447.15    |

## Prerequisition

* [DeepStream SDK 6.0 or above](https://developer.nvidia.com/deepstream-getting-started)

  Make sure deepstream-test1 sample can run successful to verify your DeepStream installation

* [tao-converter](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter/version)

  Download x86 or Jetson tao-converter which is compatible to your platform from the links in https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter/version.
* [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
 
  The LPR sample application can work as Triton client on x86 platforms.

## Download

1. Download Project with SSH or HTTPS
```
    // SSH
    git clone git@github.com:NVIDIA-AI-IOT/deepstream_lpr_app.git
    // or HTTPS
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app.git
```
2. Prepare Models and TensorRT engine

All models can be downloaded with the following commands:

```
    cd deepstream_lpr_app/
```
For US car plate recognition
```
    ./download_convert.sh us 0  #if DeepStream SDK 5.0.1, use ./download_convert.sh us 1
```
For Chinese car plate recognition
```
    ./download_convert.sh ch 0  #if DeepStream SDK 5.0.1, use ./download_convert.sh ch 1
```

## Prepare Triton Server
From DeepStream 6.1, LPR sample application supports three inferencing modes:
* gst-nvinfer inferencing based on TensorRT
* gst-nvinferserver inferencing as Triton CAPI client(only for x86)
* gst-nvinferserver inferencing as Triton gRPC client(only for x86)

The following steps are only needed for the LPR sample application working with gst-nvinferserver inferencing on x86 platforms. For LPR sample application works with nvinfer mode, please go to [Build and Run](#build-and-run) part directly.

1. Start Triton Server with DeepStream Triton container, the docker should be run in a new terminal and the following commands should be run in the same path as the deepstream_lpr_app codes are downloaded:
* For LPR sample application works as Triton CAPI client
```
    docker run --gpus all -it  --ipc=host --rm -v /tmp/.X11-unix:/tmp/.X11-unix  -v $(pwd)/deepstream_lpr_app:/lpr   -e DISPLAY=$DISPLAY -w /lpr nvcr.io/nvidia/deepstream:6.1-triton
```
* For LPR sample application works as Triton gRPC client
```
    //start Triton docker, 10001:8001 is used to map docker container's 8000 port to host's 10000 port, these ports can be changed.
    docker run --gpus all -it  --ipc=host --rm -v /tmp/.X11-unix:/tmp/.X11-unix  -p 10000:8000 -p 10001:8001 -p 10002:8002  -v $(pwd)/deepstream_lpr_app:/lpr   -e DISPLAY=$DISPLAY -w /lpr nvcr.io/nvidia/deepstream:6.1-triton

    //start tritonserver
    tritonserver --model-repository=/lpr/triton_models --strict-model-config=false --grpc-infer-allocation-pool-size=16 --log-verbose=1

    //correct Triton gRPC url, open files in deepstream-lpr-app/triton-grpc, fill the actual grpc url, like this:
    grpc {
        url: "10.23.xx.xx:10001"
    }
```

2. After the docker start to run, the following steps are needed to generate model engines inside docker:
```
    //For US car plate recognition
    ./prepare_triton_us.sh

    //For Chinese car plate recognition
    ./prepare_triton_ch.sh
```

## Build and Run
The following steps should run in host machine if the LPR sample application works as Triton client.
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
         <1: output as h264 file| 2:fakesink 3:display output> <0:ROI disable|1:ROI enable> <infer|triton|tritongrpc> \
         <input mp4 file name> ... <input mp4 file name> <output file name>
```
Or run with YAML config file.
```
    ./deepstream-lpr-app <app YAML config file>
```

### Samples

1. Application works with nvinfer

A sample of US car plate recognition:

```
    ./deepstream-lpr-app 1 2 0 infer us_car_test2.mp4 us_car_test2.mp4 output.264
```

Or run with YAML config file.
```
    ./deepstream-lpr-app lpr_app_infer_us_config.yml
```

A sample of Chinese car plate recognition:

```
    ./deepstream-lpr-app 2 2 0 infer ch_car_test.mp4 ch_car_test.mp4 output.264
 ```

2. Application works with nvinferserver(Triton native samples)

A sample of US car plate recognition:
```
    ./deepstream-lpr-app 1 2 0 triton us_car_test2.mp4 us_car_test2.mp4 output.264
```

Or run with YAML config file after modify triton part in yml file.
```
    ./deepstream-lpr-app lpr_app_triton_us_config.yml
```

A sample of Chinese car plate recognition:
```
    ./deepstream-lpr-app 2 2 0 triton us_car_test2.mp4 us_car_test2.mp4 output.264
```

Or run with YAML config file after modify triton part in yml file.
```
    ./deepstream-lpr-app lpr_app_triton_ch_config.yml
```

3. Application works with nvinferserver(Triton gRPC samples)

A sample of US car plate recognition:
```
    ./deepstream-lpr-app 1 2 0 tritongrpc us_car_test2.mp4 us_car_test2.mp4 output.264
```

Or run with YAML config file after modify triton part in yml file.
```
    ./deepstream-lpr-app lpr_app_tritongrpc_us_config.yml
```

A sample of Chinese car plate recognition:
```
    ./deepstream-lpr-app 2 2 0 tritongrpc us_car_test2.mp4 us_car_test2.mp4 output.264
```

Or run with YAML config file after modify triton part in yml file.
```
    ./deepstream-lpr-app lpr_app_tritongrpc_ch_config.yml
```

## Notice
1. This sample application only support mp4 files which contain H264 videos as input files.
2. For Chinese plate recognition, please make sure the OS supports Chinese language.
3. The second argument of deepstream-lpr-app should be 2(fakesink) for performance test.
4. The trafficcamnet and LPD models are all INT8 models, the LPR model is FP16 model.
5. There is a bug for Triton gprc mode: the first two character can't be recognized.
6. For some yolo models, some layers of the models should use FP32 precision. This is a network characteristics that the accuracy drops rapidly when maximum layers are run in INT8 precision. Please refer the [layer-device-precision](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html) for more details.

