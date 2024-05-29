## 五、TensorRT API 的基本使用

### 5.1 实战：TensorRT API 从零搭建网络

#### 5.1.1 分析从零搭建网络的优化以及流程

熟悉TensorRT API，可以先学习参考官方例程。

##### （1）TensorRT 官方例程运行

1、代码见【5.1-mnist-sample】，这里要注意的是，运行该代码需要移动文件目录：

* 把这个文件加放在你的TensorRT目录下的这个位置（把TensorRT\_DIR改成你的TensorRT安装包的位置）：

  * 这里可以学习Linux复制粘贴指令：

    * ```python
      cp -r /mnt/e/Software/LinuxOS/wsl2/tensorrt_starte
      r/chapter5-tensorrt-api-basics/5.1-mnist-sample/sampleOnnxMNIST_CN /mnt/e/Software/LinuxOS/wsl2/packages/Ten
      sorRT-8.5.1.7/samples
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/63b6665f-9298-41af-9967-5de829844587)


    * 将一个文件夹复制到另一个文件夹下：

      * ```python
        cp -r /home/packageA /home/packageB
        ```

      * 运行命令之后packageB文件夹下就有packageA文件夹了

2、之后需要在移动后的【sampleOnnxMNIST_CN】文件夹中执行```make clean```和```make```指令：

* 之后进入到$(TensorRT\_DIR)/sample/sampleOnnxMNIST\_CN/下进行make

  * ```python
    suhang@Y9000P /m/e/S/L/w/p/T/samples> cd sampleOnnxMNIST_CN/
    suhang@Y9000P /m/e/S/L/w/p/T/s/sampleOnnxMNIST_CN> ll
    total 16K
    -rwxrwxrwx 1 suhang suhang 249 May 28 09:47 Makefile
    -rwxrwxrwx 1 suhang suhang 14K May 28 09:47 sampleOnnxMNIST_CN.cpp
    
    suhang@Y9000P /m/e/S/L/w/p/T/s/sampleOnnxMNIST_CN> make clean
    
    suhang@Y9000P /m/e/S/L/w/p/T/s/sampleOnnxMNIST_CN> make
    ```

3、make之后，生成的可执行文件会在$(TensorRT\_DIR)/bin中找到：

* 在bin中执行目标文件：

  * ```py
    suhang@Y9000P /m/e/S/L/w/p/TensorRT-8.5.1.7> cd bin/
    suhang@Y9000P /m/e/S/L/w/p/T/bin> ll
    total 21M
    drwxrwxrwx 1 suhang suhang 4.0K May 28 09:50 chobj
    drwxrwxrwx 1 suhang suhang 4.0K May 28 09:49 dchobj
    -rwxrwxrwx 1 suhang suhang 2.3M Apr  1 20:37 sample_onnx_mnist
    -rwxrwxrwx 1 suhang suhang 2.3M May 28 09:50 sample_onnx_mnist_cn
    -rwxrwxrwx 1 suhang suhang 7.7M May 28 09:50 sample_onnx_mnist_cn_debug
    -rwxrwxrwx 1 suhang suhang 7.7M Apr  1 20:37 sample_onnx_mnist_debug
    -rwxrwxrwx 1 suhang suhang 581K Oct 28  2022 trtexec
    
    suhang@Y9000P /m/e/S/L/w/p/T/bin> ./sample_onnx_mnist_cn
    ```

* 整个过程是读取一个ONNX，之后Parse一个ONNX，生成一个TensorRT的engine，再做个序列化，反序列化之后把模型导出。之后读取模型，做一个前向的推理的流程
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c66cf065-f15c-43ba-ae71-5356aff8ec07)




##### （2）TensorRT 官方例程代码

1、代码见【5.1-mnist-sample】，这里要从【main()】函数：

* 流程：
  * 1. 创建一个logger用来保存日志；
  * 2. 创建sample对象，只暴露build和infer接口；
  * 3. 创建推理引擎；
  * 4. 推理。

* ```c++
  /*
   * 整个main写的比较精简。整体上通过SampleOnnxMNIST这个类把很多底层的实现部分给隐藏了
   * 我们在main中所关注的只是
   *   - "拿到一个onnx"
   *   - "parse这个onnx来生成trt推理引擎",
   *   - "推理"
   *   - "打印输出"
   * 所以程序的设计也需要把与这些不相关的不要暴露在外面。提高代码的可读性
   * 这个课程后面的代码也基本按照这个思路设计
  */
  int main(int argc, char** argv)
  {
      samplesCommon::Args args;
      bool argsOK = samplesCommon::parseArgs(args, argc, argv);
      if (!argsOK)
      {
          sample::gLogError << "Invalid arguments" << std::endl;
          printHelpInfo();
          return EXIT_FAILURE;
      }
      if (args.help)
      {
          printHelpInfo();
          return EXIT_SUCCESS;
      }
  
      // 创建一个logger用来保存日志。
      // 这里需要注意一点，日志一般都是继承nvinfer1::ILogger来实现一个自定义的。
      // 由于ILogger有一些函数都是虚函数，所以我们需要自己设计
      auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
      sample::gLogger.reportTestStart(sampleTest);
  
      // 创建sample对象，只暴露build和infer接口
      SampleOnnxMNIST sample(initializeSampleParams(args));
  
      sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;
  
      // 创建推理引擎
      if (!sample.build())
      {
          return sample::gLogger.reportFail(sampleTest);
      }
  
      // 推理
      if (!sample.infer())
      {
          return sample::gLogger.reportFail(sampleTest);
      }
  
      return sample::gLogger.reportPass(sampleTest);
  }
  ```

* 这里主要功能从创建一个logger开始，需要注意的包括三点<u>创建sample对象</u>，<u>只暴露build</u>和<u>infer接口</u>：

2、创建sample对象：

* ```c++
  /* 
   * 整个案例被封装到一个类里面了, 在类里面调用创建引擎和推理的实现
   * 这个类实现了实现的隐蔽，用户通过这个类只能调用跟推理相关的函数build, infer
   */
  class SampleOnnxMNIST
  {
  public:
      SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
          : mParams(params)
          , mEngine(nullptr)
      {
      }
      bool build();
      bool infer();
  
  private:
      samplesCommon::OnnxSampleParams mParams; 
      nvinfer1::Dims                  mInputDims;  
      nvinfer1::Dims                  mOutputDims;
      int mNumber{0};         
  
      /* 使用智能指针来指向引擎，方便生命周期管理 */
      std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
  
      /* 创建网络 */
      bool constructNetwork(
          SampleUniquePtr<nvinfer1::IBuilder>& builder,
          SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
          SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
          SampleUniquePtr<nvonnxparser::IParser>& parser);
  
      bool processInput(const samplesCommon::BufferManager& buffers);
      bool verifyOutput(const samplesCommon::BufferManager& buffers);
  };
  ```

* 这里用到了智能指针，可以不用再free了

3、创建网络build()：

* 创建网络的流程基本上是这样：
  * 1. 创建一个builder
    2. 通过builder创建一个network
    3. 通过builder创建一个config
    4. 通过config创建一个opt(这个案例中没有)
    5. 对network进行创建
    6. 可以使用parser直接将onnx中各个layer转换为trt能够识别的layer (这个案例中使用的是这个)
    7. 也可以通过trt提供的ILayer相关的API自己从零搭建network (后面会讲)
    8. 序列化引擎(这个案例中没有)
    9. Free(如果使用的是智能指针的话，可以省去这一步)

4、推理的实现部分infer()：

* 推理的实现部分。注意这里面把反序列化的部分给省去了。直接从创建context开始
  * context就是上下文，用来创建一些空间来存储一些中间值。通过engine来创建
  * 一个engine可以创建多个context，用来负责多个不同的推理任务
  * 另外context可以复用。也就是每次新的推理可以利用之前创建好的context
* 整个流程如下
  * 1. 创建context
    2. 对于MNIST数据的preprocess(预处理)部分, 这个案例是在CPU上实现的
    3. 将host上预处理好的数据copy到device上
    4. 进行TensorRT的forward推理实现
       1. 创建好了context之后，推理只需要使用executeV2或者enqueueV2就可以了
       2. 之后trt会自动根据创建好的engine来逐层进行forward
    5. 将device上forward好的数据copy到host上
    6. postprocess(后处理的实现)



学习完参考官方例程的搭建网络优化流程后，开始进行自己创建网络的实验。

##### （3）build-model

执行这个函数要在终端中先`make clean`一下之前生成的内容再`make`，之后删除`【models文件】`夹中`【engine文件】`夹中的`sample.engine`的模型，最后便可以运行可执行文件`./trt-infer`了：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c6bb5ce8-87de-4b0c-85d1-18064ee7f95e)

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bbffda12-8829-4914-938c-2864a76a15ff)


下面针对代码进行学习，代码见【5.2-load-model】

**1、主函数**

主函数隐藏了细节，为用户提供了函数接口方便创建

* ```c++
  #include <iostream>
  #include <memory>
  
  #include "model.hpp"
  #include "utils.hpp"
  
  using namespace std;
  
  int main(int argc, char const *argv[])
  {
      Model model("models/onnx/sample.onnx");
      if(!model.build()){
          LOGE("ERROR: fail in building model");
          return 0;
      }
      return 0;
  }
  
  ```

* 用`Model`类的构造函数创建了一个`model`，这是main函数中的主要功能，那么就要继续分析`Model`这个类的作用

**2、Model 类**

`Model`类中包含了公有接口和私有接口：

* ```c++
  class Model{
  public:
      Model(std::string onnxPath);
      bool build();
  private:
      std::string mOnnxPath;
      std::string mEnginePath;
      nvinfer1::Dims mInputDims;
      nvinfer1::Dims mOutputDims;
      std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
      bool constructNetwork();
      bool preprocess();
  };
  ```

* 公有接口包含<u>Model的构造函数</u>和<u>build函数</u>

**3、Model 的构造函数**

Model的构造函数主要是三个功能：1.查看model文件路径"onnxPath"是否存在、2.创建mOnnxPath、3.创建mEnginePath

* ```c++
  Model::Model(string onnxPath){
      // 查看model文件路径"onnxPath"是否存在
      if (!fileExists(onnxPath)) {
          LOGE("%s not found. Program terminated", onnxPath.c_str());
          exit(1);
      }
      // 创建mOnnxPath和mEnginePath
      mOnnxPath   = onnxPath;
      mEnginePath = getEnginePath(mOnnxPath);
  }
  ```

**4、build() 函数**

Model的构造函数主要是三个功能：1.查看文件路径"mEnginePath"是否存在，2.创建Logger，3.创建builder、network、config、parser，parse，序列化，反序列化等：

* ```c++
  bool Model::build(){
      // 查看文件路径"mEnginePath"是否存在
      if (fileExists(mEnginePath)){
          LOG("%s has been generated!", mEnginePath.c_str());
          return true;
      } else {
          LOG("%s not found. Building engine...", mEnginePath.c_str());
      }
      // 创建一个Logger
      // 自己创建的logger需要继承ILogger，并实现log虚函数
      Logger logger;
      // 这里从logger中创建builder
      auto builder       = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
      // 这里从builder中创建network
      auto network       = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(1));
      // 这里从builder中创建config
      auto config        = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
      // 这里从network和logger中创建parser
      auto parser        = make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
  
      // 从config中创建一些跟模型有关的信息
      config->setMaxWorkspaceSize(1<<28);
  
      // 从文件里parse放到network里面去
      if (!parser->parseFromFile(mOnnxPath.c_str(), 1)){
          LOGE("ERROR: failed to %s", mOnnxPath.c_str());
          return false;
      }
  
      // 从builder中创建engine，这里需要传入*network, *config两个参数
      auto engine        = make_unique<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
      // 从builder中对创建好的模型做一个序列化，并保存到plan中去
      auto plan          = builder->buildSerializedNetwork(*network, *config);
      // 创建反序列化引擎的runtime
      auto runtime       = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  
      // 打开文件，把plan->data(), plan->size()信息写入
      auto f = fopen(mEnginePath.c_str(), "wb");
      fwrite(plan->data(), 1, plan->size(), f);
      fclose(f);
  
      // 用runtime做反序列化，生成的指针保存到mEngine中
      mEngine            = shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
      // 从networt中打印出input和output信息
      mInputDims         = network->getInput(0)->getDimensions();
      mOutputDims        = network->getOutput(0)->getDimensions();
      LOG("Input dim is %s", printDims(mInputDims).c_str());
      LOG("Output dim is %s", printDims(mOutputDims).c_str());
      return true;
  };
  
  ```

* 需要注意的点：

  * 1. 自己创建的logger需要继承ILogger，并实现log虚函数:

    * ```c++
      //自己创建的logger需要继承ILogger,并实现log虚函数
      //这里Logger继承了nvinfer1的ILogger
      class Logger : public nvinfer1::ILogger{
      public:
          virtual void log (Severity severity, const char* msg) noexcept override{
              string str;
              switch (severity){
                  case Severity::kINTERNAL_ERROR: str = RED    "[fatal]:" CLEAR;
                  case Severity::kERROR:          str = RED    "[error]:" CLEAR;
                  case Severity::kWARNING:        str = BLUE   "[warn]:"  CLEAR;
                  case Severity::kINFO:           str = YELLOW "[info]:"  CLEAR;
                  case Severity::kVERBOSE:        str = PURPLE "[verb]:"  CLEAR;
              }
              // 打印Severity::kINFO级别以下的信息
              if (severity <= Severity::kINFO)
                  cout << str << string(msg) << endl;
          }
      };
      ```

    * 这里的virtual虚函数是对ILogger内部的虚函数的重写，设置了安全等级severity，能够打印等级内的信息。

  * 2. 可用从config中创建一些和模型有关的东西：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e26f61fa-f6dc-4e5f-b315-334434aa3354)

  * 3. TensorRT C++ API 可以谷歌直接搜索找到：
       1. [Developer Guide :: NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#overview)

* 创建后的模型结构展示：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/57613da7-d9a8-42e2-add2-b2ce2c5d8cdf)




##### （4）infer-model

20240530更新




