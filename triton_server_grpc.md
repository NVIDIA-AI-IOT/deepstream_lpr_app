# Triton Server
## [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) Bring Up

DeepStream applications can work as Triton Inference client. So the corresponding Triton Inference Server should be started before the Triton client start to work.

A immediate way to start a corresponding Triton Server is to use Triton containers provided in [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver). Since every DeepStream version has it corresponding Triton Server version, so a reliable way is to use the [DeepStream Triton container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream).

* The Triton Server can be started in the same machine which the DeepStream application works in, please make sure the Triton Server is started in a new terminal.

* The Triton Server can be started in another machine as the server, the necessary scripts and configuration files are provided in the LPR sample repository, so the project should be downloaded in the server machine as mentioned in the README "Download" part. 

## Prepare Triton Server For gRPC Connection
The following steps take the DeepStream 6.1 GA as an example, if you use other DeepStream versions, the corresponding DeepStream Triton [image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) can be used.

To start Triton Server with DeepStream Triton container, the docker should be run in a new terminal and the following commands should be run in the same path as the deepstream_lpr_app codes are downloaded:
* For LPR sample application works as Triton gRPC client
```
    //start Triton docker, 10001:8001 is used to map docker container's 8000 port to host's 10000 port, these ports can be changed.
    docker run --gpus all -it  --ipc=host --rm -v /tmp/.X11-unix:/tmp/.X11-unix  -p 10000:8000 -p 10001:8001 -p 10002:8002  -v $(pwd)/deepstream_lpr_app:/lpr   -e DISPLAY=$DISPLAY -w /lpr nvcr.io/nvidia/deepstream:6.1-triton
```

Then the model engines should be generated in the server

```
    //For US car plate recognition
    ./prepare_triton_us.sh

    //For Chinese car plate recognition
    ./prepare_triton_ch.sh
```

If the server is running in the same machine as the DeepStream application, the following command can be used directly. If it is not, please set the gRPC url as the IP address of the server machine in the following configuration files in deepstream-lpr-app/triton-grpc folder:
* lpr_ch_config.txt
* pgie_config.txt
* lpd_DetectNet2_us.txt
* lpd_yolov4-tiny_ch.txt
* lpd_DetectNet2_ch.txt
* lpd_yolov4-tiny_us.txt
* lpr_us_config.txt

The gRPC url setting looks like:
```
grpc {
        url: "10.23.xx.xx:10001"
    }
```

Then the Triton Server service can be started with the following command:
```
    //start tritonserver for US plate recognition
    tritonserver --model-repository=/lpr/triton_models/us --strict-model-config=false --grpc-infer-allocation-pool-size=16 --log-verbose=1

    //start tritonserver for Chines plate recognition
    tritonserver --model-repository=/lpr/triton_models/ch --strict-model-config=false --grpc-infer-allocation-pool-size=16 --log-verbose=1
```

The LPR sample application should run in another terminal with Triton Inference client libraries installed. It is recommend to run the application in the DeepStream Triton container, please refer to [triton_server.md](https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app/blob/master/triton_server.md) for how to start a DeepStream Triton container.
