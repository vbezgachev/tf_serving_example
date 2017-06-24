## Install Docker
- Download DEB package from https://download.docker.com/linux/ubuntu/dists/xenial/pool/stable/amd64/ to, e.g., ~/Downloads
- Install Docker
```
cd ~/Downloads
sudo dpkg -i <package name>.deb
```
- Add current user to Docker group. This allows to execute docker command without sudo
```
sudo usermod -a -G docker $USER
```
- Log out and login again into the system. Open terminal and type
```
docker version
```
You should see a docker version information without any permissions issue

## Create docker image
- Change to your working directory
```
cd <path to your working directory>
```
- Clone the sources from the git (takes up to 5 minutes depending on Internet connection)
```
git clone --recurse-submodules https://github.com/tensorflow/serving
```

### CPU build of TensorFlow
- Create Docker container (takes up to 10 minutes to download all necessary stuff and build everything)
```
cd serving
docker build --pull -t $USER/tensorflow-serving-devel -f tensorflow_serving/tools/docker/Dockerfile.devel .
```
- Run docker container in interactive mode
```
docker run --name=inception_container -it $USER/tensorflow-serving-devel
```
- Install _vim_ for future use:
```
apt-get update
apt-get install vim
```
- If you restart the system, you should not execute the run command, instead you should start the existing docker container:
```
docker start -i inception_container
```

### GPU build of TensorFlow
**CAUTION**
```
docker build --pull -t $USER/tensorflow-serving-devel-gpu -f tensorflow_serving/tools/docker/Dockerfile.devel-gpu . 
```
does not work, see [https://github.com/tensorflow/serving/issues/327](https://github.com/tensorflow/serving/issues/327).  
  
**Workaround**
- Edit tensorflow_serving/tools/docker/Dockerfile.devel-gpu
  * Create a symbolic link to paths in 
  ```
  RUN mkdir /usr/lib/x86_64-linux-gnu/include/ && \
  ```
  group of commands:
   ```
  ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
  ```
  * Comment out or remove
  ```
  WORKDIR
  ```
  and
  ```
  RUN bazel_build
  ```
- Run Docker container in interactive mode: 
```
docker run --name=inception_container_gpu -it $USER/tensorflow-serving-devel-gpu
cd /serving/tensorflow/tensorflow/contrib/
```
- Install _vim_:
```
apt-get update
apt-get install vim
```
Now we need to change sources to build TensorFlow with GPU support
- Edit BUILD file
```
vim BUILD
```
- Scroll down to the dependency
```
//tensorflow/contrib/nccl:nccl_py
```
and comment it out
- Edit [nccl](https://github.com/NVIDIA/nccl) stuff
  * _nccl_manager.h_
  ```
  cd /serving/tensorflow/tensorflow/contrib/nccl/kernels
  vim nccl_manager.h
  ```
  Change
  ```
  #include "external/nccl_archive/src/nccl.h"
  ```
  to
  ```
  #include "src/nccl.h"
  ```
  * _nccl_ops.cc_
  ```
  vim nccl_ops.cc
  ```
  Do the same operation as previously
- If you restart the system, you should not execute the run command, instead you should start the existing docker container:
```
docker start -i inception_container
```

## Build and try TensorFlow serving in docker container
The following operations are the same for CPU and GPU builds.
- Clone, configure and build Serving in the container
```
cd ~
git clone --recurse-submodules https://github.com/tensorflow/serving
Now we can build tensorflow
cd serving/tensorflow
./configure
```
Accept all defaults.

- Build TensorFlow serving
```
cd ..
bazel build -c opt tensorflow_serving/…
```
Wait 30 – 40 minutes
After successful build you should be able to execute the following statement:
```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```
You should see now the usage documentation

## Deploy the model
- Download the model
```
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
tar xzf inception-v3-2016-03-01.tar.gz
```
- Export the model
```
bazel-bin/tensorflow_serving/example/inception_export --checkpoint_dir=inception-v3 –export_dir=inception-export
```
- As a result you should see
```
Successfully loaded model from inception-v3/model.ckpt-157585 at step=157585.
Exporting trained model to inception-export/1
Successfully exported model to inception-export
```

## Test the functioning
- Start the gRPC server
```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception --model_base_path=inception-export &> inception_log &
```
- Download test image
```
apt-get update
apt-get install wget
wget https://upload.wikimedia.org/wikipedia/en/a/ac/Xiang_Xiang_panda.jpg
```
- Try
```
bazel-bin/tensorflow_serving/example/inception_client --server=localhost:9000 –image=./Xiang_Xiang_panda.jpg
```
- In case of a timeout issue
```
cd bazel-bin/tensorflow_serving/example/inception_client.runfiles/tf_serving/tensorflow_serving/example
vim inception_client.py
```
Scroll to 
```
result = stub.Predict(request, 10.0)  # change this value 
```
and change timeout value
- Now the output should be
```
outputs {
key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    string_val: "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca"
    string_val: "indri, indris, Indri indri, Indri brevicaudatus"
    string_val: "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens"
    string_val: "gibbon, Hylobates lar"
    string_val: "sloth bear, Melursus ursinus, Ursus ursinus"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    float_val: 8.98223686218
    float_val: 5.39600038528
    float_val: 5.00718212128
    float_val: 2.93680524826
    float_val: 2.78477811813
  }
}
```

## Train your model
**Use TensorFlow 1.1 (not 1.2)!**

## Export your model

## Copy exported model into the Docker
- Go to the folder with exported model
- Copy it into the Docker container
```
docker cp ./model-export inception_container:/serving
```

## Execute client locally to call the server in the Docker
- We need to install grpcio, grpcio-tools in our python environment
- Generate Python code from protobuf
  * Rename serving/tensorflow to e.g. serving/tensorflow_
  * Copy serving/tensorflow_/tensorflow to serving (i.e. move it one level up). That ensures that imports in proto files work correctly
  * Execute
  ```
  python -m grpc.tools.protoc /home/vetal/Work/dl/serving/tensorflow_serving/apis/*.proto --python_out=./tf_deploy --grpc_python_out=./tf_deploy --proto_path=/home/vetal/Work/dl/serving
  ```