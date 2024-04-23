### 3.1 TensorRT简介

“NVIDIA TensorR, an **SDK** for **high-performance** deep learning **inference**, includes a deep  learning inference optimizer and runtime that delivers low latency and high throughput for  inference applications.”

* 可以把TensorRT理解为一个针对N卡GPU的编译优器，它可以：
  * 自动优化模型
    * 寻找模型中可以并行处理的地方
    * 针对当前部署的GPU框架，寻找最优的调度和并行策略
  * 支持多框架输入
    * ONNX
  * Python/C++ API接口
    * 可以方便在自己的程序中调用TensorRT API来实现推理

#### 3.1.1 TensorRT的工作流介绍

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/51bd2854-872a-437a-8426-42609468d4dc)

#### 3.1.2 TensorRT的一些限制

#### （1）针对不支持的算子

* 看看不同TensorRT版本中是否有做了支持
  * PyTorch: Facebook
  * ONNX: Microsoft
  * TensorRT: NVIDIA
* 解决办法：
  * 修改PyTorch的算子选择，让它使用一些TensorRT支持的算子
  * 自己写插件，内部实现自定义算子以及自定义cuda加速核函数
  * 不使用onnx，自己创建parser直接调用TensorRTAPI逐层创建网络
    * 比如darknet模型

#### （2）不同TensorRT版本的优化策略是不一样的

* 比如：对Transformer的优化和TensorRT 7和8跑出来的性能是不一样的

#### （3）有时你预期TensorRT的优化和实际的优化是不一样的

* 比如：你期望它使用Tensor core，但kernel autotuning之后TensorRT觉得使用Tensor core反而会效率降低，结果给你分配CUDA core使用。因为内部需要做一些额外的处理。

#### （4）天生并行性差的layer，TensorRT也没有办法

* 1x1 conv这种layer，再怎么优化也没有7x7 conv并行效果好

### 3.2 TensorRT的应用场景

#### 3.2.1 TensorRT的优化意义

##### （1）模型训练

* 需要关注的点：
  * 把模型做的深一点，宽一点
  * 使用丰富的Data augmentation
  * 使用各种Training trick

##### （2）模型部署

* 以精度不变，或者精度掉点很小的情况下尽量压缩模型：
  * 减少计算量
  * 减少memory access
  * 提高计算密度

#### 3.2.2 自动驾驶

##### （1）需要关注的散热性

* 常规的GPU有大又耗电，在自动驾驶车上又要性能高，又要不那么热，那么很多地方就会把视线集中在Jetson系列上

##### （2）自动驾驶需要关注的实时性

* 实时性是我们在部署模型是需要关注的一个很重要的指标：
  * 一般来说15 fps是一个底线 (高速公路上需要至少30fps)
  * 100km/h = 28m/s
* 对于单个小型DNN，在计算资源限制的情况下想达到这个指标难度系数不高
  * 模型可以做的很simple
* 但我们又得需要看的很远
  * 比如说150m远的红绿灯，以及红绿灯下面的箭头和时间
  * 比如说100m远的路标
*  不光是Perception，后期的Planning等等也需要时间
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/295f9451-a2b7-4a51-9744-5047612a623c)

  * 如果把这些所有的事情能够在几短时间内做完并达到>10fps，甚至30fps， 是一个非常具有挑战性的事情
* 工业界提高实时性办法：
  * 1、局部提高分辨率 优先处理局部区域：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f3ad55be-be47-4aa1-b481-e4ddf13082f3)

  * 2、CPU与GPU异步处理

##### （3）自动驾驶中需要关注的电力消耗

* 基本上我们需要的算力越大，消耗的电力越大。自动驾驶所需要控制的电力一般来说需要控制在小于100w。自动驾驶也属 于Edge computing的一种，所以对电力消耗需要严格控制。
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f9bf93dc-4313-4b36-824a-ba348291db73)


##### （4）自动驾驶中模型部署所关注的东西

* RealTime(实时性)
* Power Consumption(消耗电力)
* Long range Accuracy(远距离) 



### 3.3 TensorRT的模块

#### 3.3.1 TensorRT的优化策略

##### （1）Layer fusion-层融合

* 层融合种类：
  * Vertical layer fusion (垂直层融合)
    * 用的比较常见，针对conv + BN + ReLU进行融合
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6e3c22a9-16bf-44b0-96fe-198fffaa7347)

  * Horizontal layer fusion (水平层融合)
    * 当模型中有水平方向上比较多的同类layer，会直接进行融合
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e399aad6-e244-4161-a98d-31d526f2cc0a)

* 层融合可以减少启动kernel的开销与memory操作，从而提高效率。同时，有些计算可以通过层融合优化后，跟其他计算合并
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a6320c29-8c2e-44ed-aea6-77b0fee74cd7)


##### （2）Kernel auto-tuning

* TensorRT内部对于同一个层使用各种不同kernel函数进行性能测试
  * 比如对于FC层中的矩阵乘法，根据tile size有很多中kernel function
  * (e.g. 32x32, 32x64, 64x64, 64x128, 128x128，针对不同硬件有不同策略)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1f297cbe-8c23-45b7-8d6e-0faaf7328f57)


##### （3）Quantization-量化

* 量化是压缩模型的一个很重要的策略
  * 将单精度类型(FP32)训练权重转变为半精度(FP16)或者整型(INT8, INT4)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/383c3203-1203-477e-8458-a2731690ee43)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1f81885b-7936-4e41-97ef-db8694f7fa1a)

* 过程作量化(Quantization)：
  * 训练的时候，因为需要**优先考虑精度**而不需要太重视速度，所以会使用 FP32来表示权重和激活值；
  * 但部署的时候，我们需要想办法把FP32的数据尽量压缩，能够用**16bits,  8bits**，甚至**4 bits**来表示它们。
  *  ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8cb223c1-e089-4900-8dae-c92197cd9f35)


### 3.4 导出ONNX以及修改ONNX的方法

### 3.5 初步使用TensorRT
