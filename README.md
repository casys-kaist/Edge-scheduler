# EdgeScheduler

## System Requirements
```
Ubuntu 14.04, the latest Android SDK, the latest Android NDK, Caffe.  
```

## Install the SNPE SDK (Snapdragon Neural Processing Engine)
```
Please refer to https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/getting-started
```

## Add EdgeScheduler to the SNPE framework
```
$ cp main.cpp ~/snpe-sdk/examples/NativeCpp/SampleCode/jni/main.cpp
```

## Build SNPE frame
```
$ cd ~/snpe-sdk/
$ export ANDROID_NDK_ROOT=~/Android/Sdk/ndk-bundle # default location for Android Studio, replace with yours
$ source ./bin/envsetup.sh -c ~/caffe
Please refer to https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/getting-started
```

