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
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8cb223c1-e089-4900-8dae-c92197cd9f35)


### 3.4 导出ONNX以及修改ONNX的方法

#### 3.4.1 报错

##### （1）未安装Pytorch

* 会显示【ModuleNotFoundError: No module named ‘torch’】

* 原因是没有安装pytorch

  * 因为我们配置的是CUDA11.7，所以在下面网页原则相对应的版本pip即可：

    * [Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/53ace0e4-579f-4809-b597-66e8896fcc6a)


    * ```python
      # CUDA 11.7
      pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
      ```

##### （2）未找到models路径

* 会显示【FileNotFoundError: [Errno 2] No such file or directory: '../models/example.onnx'】
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/cb0c720b-f592-4812-ad74-ce25d684f1e4)

* 原因是没有创建【models】文件夹
  * 新建即可
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4137ca80-2589-4ece-a9e5-0919a991852b)


#### 3.4.2 导出ONNX

##### （1）程序代码

* ```python
  import torch
  import torch.nn as nn
  import torch.onnx
  
  class Model(torch.nn.Module):
      def __init__(self, in_features, out_features, weights, bias=False):
          super().__init__()
          self.linear = nn.Linear(in_features, out_features, bias)
          with torch.no_grad():
              self.linear.weight.copy_(weights)
      
      def forward(self, x):
          x = self.linear(x)
          return x
  
  def infer():
      in_features = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
      weights = torch.tensor([
          [1, 2, 3, 4],
          [2, 3, 4, 5],
          [3, 4, 5, 6]
      ],dtype=torch.float32)
      
      model = Model(4, 3, weights)
      x = model(in_features)
      print("result is: ", x)
  
  def export_onnx():
      input   = torch.zeros(1, 1, 1, 4)
      weights = torch.tensor([
          [1, 2, 3, 4],
          [2, 3, 4, 5],
          [3, 4, 5, 6]
      ],dtype=torch.float32)
      model   = Model(4, 3, weights)
      model.eval() #添加eval防止权重继续更新
  
      # pytorch导出onnx的方式，参数有很多，也可以支持动态size
      # 我们先做一些最基本的导出，从netron学习一下导出的onnx都有那些东西
      # opset：指令集版本
      torch.onnx.export(
          model         = model, 
          args          = (input,),
          f             = "../models/example.onnx",
          input_names   = ["input0"],
          output_names  = ["output0"],
          opset_version = 12)
      print("Finished onnx export")
  
  
  if __name__ == "__main__":
      infer()
      export_onnx()
  ```

* 1、首先进入【if __name__ == "__main__":】

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/189293ab-2438-4212-beb7-a490958a7a08)


* 2、然后运行【infer()】函数

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ffb58da4-bfec-499f-a7ae-cd9d911ccb9c)

  * 其中函数会调用【Model】：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8aff615c-9d1b-492c-9261-62551af42b51)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e7d5f66f-e611-482f-b014-077bf3641fbf)


* 3、最后运行【export_onnx()】函数：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/33e49cf6-5723-4a33-b0a6-778164cd205b)




##### （2）查看流程

* 1、运行代码：

  * ```python
    python3 example.py
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/585311eb-bd56-40bb-8643-a1ad6f7d80be)


* 2、使用netron导出结构图：

  * ```python
    netron ../models/example.onnx 
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/afdcfe55-2784-49ef-ba59-004a3164efc9)


  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/67721f79-c193-4342-9b1a-c1f5c6f2dc44)


##### （3）双输出

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3be1b718-5597-4a4f-a869-06ad29e63d57)

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c273afce-aa08-4515-90b2-58e0d1cc1cbe)

* 代码中需要补充的地方：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4c84f95d-e91a-4744-b990-95441b29c11c)


##### （4）动态batch

* 输出：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6502c321-61a8-44e4-9a7c-a40b0c10e563)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/072bcc09-22b0-4ee1-b59f-6ffcdaa0e21c)

* 代码细节：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0c0ce173-04b9-454e-a486-b3ccb29eb98e)




#### 3.4.3 偏向深度学习的onnx框架

##### （1）cbr框架

* 1、框架结构：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/336a28d0-2867-4bc9-9452-b484a10e890f)

* 2、代码细节：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/19d2275f-0c09-4e4e-b325-1ecdc662c229)

  * 由于代码中给定了BatchNorm2d，但是在框架中并没有显示，是因为在onnx导出时，已经将BarchNorm2d和Conv2d融合在了一起，所以不会显示。

##### （2）reshape框架

* 

### 3.5 初步使用TensorRT
