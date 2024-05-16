### 3.1 TensorRT简介十大

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

##### （1）cbr

* 1、框架结构：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bc1d008e-5ad7-4e78-a877-1ad1b8049b09)

* 2、代码细节：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a040896b-a2ef-4a3c-829d-9c24d91235ab)

  * 由于代码中给定了BatchNorm2d，但是在框架中并没有显示，是因为在onnx导出时，已经将BarchNorm2d和Conv2d融合在了一起，所以不会显示。

##### （2）reshape

* 1、框架结构：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/989780b4-a4eb-4d24-8bd2-180ffcabfc7e)

* 2、代码细节：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/589f9cc4-db64-499c-8ade-9a295b2fb3c5)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/de121316-013a-4315-b9e5-226da9cdabf2)

* 3、可以使用onnx-simplifier来进行onnx的简化（偏工业）：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0deb3e9c-5cc8-494b-9669-a46ecf864813)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/eccdac24-37b5-4c62-af42-ee35997d16a1)


##### （3）导出torchvision中提供的框架

* 1、代码细节：
  * 设置模型选择选项：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3f67c99b-f623-4d33-9396-211fcd08aaa7)

  * 指定导出模型：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b3305c0c-6529-46ac-9bc5-c5fcafe9ecd8)

* 2、模型结构
  * 可以导出自行学习，模型较大，这里不再展示。

#### 3.4.4 剖析onnx架构并理解ProtoBuf

##### （1）ONNX简介

* ONNX是一种神经网络的格式，采用Protobuf二进制形式进行序列化模型。
  * Protobuf: 全称叫做Protocal Buffer。是Google提出来的一套表示和序列化数据的机制
  * Protobuf会根据用于定义的数据结构来进行序列化存储
* 同理，我们可以根据官方提供的数据结构信息，去修改或者创建onnx
  * 查看网址：
    * [onnx/onnx/onnx.in.proto at main · onnx/onnx (github.com)](https://github.com/onnx/onnx/blob/main/onnx/onnx.in.proto)
    * 里面包含了onnx的protobuf数据格式
* onnx中的组织架构：
  * ModelProto（描述的是整个模型的信息）
    * 一般用来定义模型的全局信息，比如opset
      * (graph并不是repeated，所以一个model对应一个graph)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ee4e17d3-8357-4a0d-a654-a0ae388c7af6)

    * GraphProto（描述的是整个网络的信息）
      * 一般用来定义一个网络。包括
        * input/output(input/output是repeated，所以是数组)
        * initializer (initializer是repeated，所以是数组)
        * node (node是repeated，所以是数组) 
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8cfb4c6d-5846-4d42-830b-4aaa5b1dd4a1)

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/745a6ed1-e97d-4c3e-b927-3dd2ac3d417d)

      * NodeProto（描述的是各个计算节点，比如conv，liner）
        * 一般用来定义一个计算节点，比如conv, linear
          * (input是repeated类型，意味着是数组) 
          * (output是repeated类型，意味着是数组) 
          * (attribute有一个自己的Proto) 
            * 一般用来定义一个node的属性。比如说kernel size
              *  ( 比较常见的方式就是把(key, value)传入Proto，之后 
                * name = key
                * i         = value 
              * )
            * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ba2f4f20-6230-42f9-9f3d-a0769a5e28fd)

            * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/89088ba8-f013-41ca-b099-f42e1bd23706)

          * (op_type需要严格根据onnx所提供的Operators写)
            *  https://github.com/onnx/onnx/blob/main/docs/Operators.md
            * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ccff9bdb-0a41-4b57-a435-255d38f29dcb)

      * TensorProto（描述的是tensor的信息，主要包括权重）
        * 一般用来定义一个权重，比如conv的w和b
          * (dims是repeated类型，意味着是数组) 
          * (raw_data是bytes类型)
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/731c502d-4dbb-41d3-87bc-e93ae485522a)

      * ValueInfoProto（描述的是input/output信息）
        * 一般用来定义网络的input/output 
          * (会根据input/output的type来附加属性)
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8ea8c057-a244-43c4-9687-a707ce00198f)


##### （2）创建Linear网络

* Linear网络可参考4个算子的运算：

  * y = (a * x) + b
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9ec794ff-9cf0-4736-be1a-e7a33656d76e)


* 1、要将上述4个算子运算，需先创建4个算子的数据结构，因此分别对【a，x，b，y】创建 TensorProto（也是ValueProto）

  * ```python
    def create_onnx():
        # 创建ValueProto
        a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
        x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
        b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
        y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 10])
    ```

* 2、创建Mul和Add两个运算节点，在NodeProto中创建：

  * ```python
        # 创建NodeProto
        mul = helper.make_node('Mul', ['a', 'x'], 'c', "multiply")
        add = helper.make_node('Add', ['c', 'b'], 'y', "add")
    ```

  * 注意：输入的第一个参数要与opset列表内容一致：

    * [onnx/docs/Operators.md at main · onnx/onnx (github.com)](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1716eb81-0d48-4b85-945f-7251b5538163)


* 3、创建节点连接的网络：

  * ```python
        # 创建NodeProto
        graph = helper.make_graph([mul, add], 'sample-linear', [a, x, b], [y])
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/38139894-8d80-4ff5-b50c-f4e0f732dc55)


* 4、创建Model，只需要将graph参数传入即可：

  * ```python
        # 构建ModelProto
        model = helper.make_model(graph)
    ```

* 5、检查Model，可用print打印

  * ```python
        # 检查model是否有错误
        onnx.checker.check_model(model)
        # print(model)
    ```

* 6、保存model

  * ```python
    	onnx.save(model, "../models/sample-linear.onnx")
    return model
    ```



##### （3）创建Convnet网络

* 1、网络结构：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/002f00ff-e60c-409a-a48a-5ad6e221762c)


* 2、给定全局参数：

  * ```python
    def main():
        
        input_batch    = 1;
        input_channel  = 3;
        input_height   = 64;
        input_width    = 64;
        output_channel = 16;
    
        input_shape    = [input_batch, input_channel, input_height, input_width]
        output_shape   = [input_batch, output_channel, 1, 1]
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c8124291-f9a2-4e0b-b2d5-8ce76d7a6204)


    * input_shape：[B, C, H, W]
    * output_shape：[B, C, 1, 1]

* 3、创建Input，Output参数，类型为TensorProto：

  * ```python
        ##########################创建input/output################################
        model_input_name  = "input0"
        model_output_name = "output0"
    
        input = onnx.helper.make_tensor_value_info(
                model_input_name,
                onnx.TensorProto.FLOAT,
                input_shape)
    
        output = onnx.helper.make_tensor_value_info(
                model_output_name, 
                onnx.TensorProto.FLOAT, 
                output_shape)
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3be7b34a-2e79-4f3b-8279-a9861bfea58e)


* 4、创建第1个Conv节点：

  * ```python
        ##########################创建第一个conv节点##############################
        conv1_output_name = "conv2d_1.output"
        conv1_in_ch       = input_channel
        conv1_out_ch      = 32
        conv1_kernel      = 3
        conv1_pads        = 1
    
        # 创建conv节点的权重信息
        conv1_weight    = np.random.rand(conv1_out_ch, conv1_in_ch, conv1_kernel, conv1_kernel)
        conv1_bias      = np.random.rand(conv1_out_ch)
    
        conv1_weight_name = "conv2d_1.weight"
        conv1_weight_initializer = create_initializer_tensor(
            name         = conv1_weight_name,
            tensor_array = conv1_weight,
            data_type    = onnx.TensorProto.FLOAT)
    
    
        conv1_bias_name  = "conv2d_1.bias"
        conv1_bias_initializer = create_initializer_tensor(
            name         = conv1_bias_name,
            tensor_array = conv1_bias,
            data_type    = onnx.TensorProto.FLOAT)
    
        # 创建conv节点，注意conv节点的输入有3个: input, w, b
        conv1_node = onnx.helper.make_node(
            name         = "conv2d_1",
            op_type      = "Conv",
            inputs       = [
                model_input_name, 
                conv1_weight_name,
                conv1_bias_name
            ],
            outputs      = [conv1_output_name],
            kernel_shape = [conv1_kernel, conv1_kernel],
            pads         = [conv1_pads, conv1_pads, conv1_pads, conv1_pads],
        )
    
    ```

  * 创建Conv节点输入量有3个：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4d83a4e8-1b5d-48c1-beea-0d8e196ea627)

      * Input（上一个）；
      * Conv1_weight；
      * Conv1_bias。

  * 其中Conv1_weight 和 Conv1_bais 需要用 Initializer 创建各自的权重：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d0682fc2-c3e3-4a32-a090-cc9e19f52e8f)


  * 输出除了outputs，还要分配 kernel_shape 和 pads：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/fb81f542-f0b0-48ca-a0c2-efee08f75abc)


* 5、创建第1个BatchNorm节点：

  * ```python
        ##########################创建一个BatchNorm节点###########################
        bn1_output_name = "batchNorm1.output"
    
        # 为BN节点添加权重信息
        bn1_scale = np.random.rand(conv1_out_ch)
        bn1_bias  = np.random.rand(conv1_out_ch)
        bn1_mean  = np.random.rand(conv1_out_ch)
        bn1_var   = np.random.rand(conv1_out_ch)
    
        # 通过create_initializer_tensor创建权重，方法和创建conv节点一样
        bn1_scale_name = "batchNorm1.scale"
        bn1_bias_name  = "batchNorm1.bias"
        bn1_mean_name  = "batchNorm1.mean"
        bn1_var_name   = "batchNorm1.var"
    
        bn1_scale_initializer = create_initializer_tensor(
            name         = bn1_scale_name,
            tensor_array = bn1_scale,
            data_type    = onnx.TensorProto.FLOAT)
        bn1_bias_initializer = create_initializer_tensor(
            name         = bn1_bias_name,
            tensor_array = bn1_bias,
            data_type    = onnx.TensorProto.FLOAT)
        bn1_mean_initializer = create_initializer_tensor(
            name         = bn1_mean_name,
            tensor_array = bn1_mean,
            data_type    = onnx.TensorProto.FLOAT)
        bn1_var_initializer  = create_initializer_tensor(
            name         = bn1_var_name,
            tensor_array = bn1_var,
            data_type    = onnx.TensorProto.FLOAT)
    
        # 创建BN节点，注意BN节点的输入信息有5个: input, scale, bias, mean, var
        bn1_node = onnx.helper.make_node(
            name    = "batchNorm1",
            op_type = "BatchNormalization",
            inputs  = [
                conv1_output_name,
                bn1_scale_name,
                bn1_bias_name,
                bn1_mean_name,
                bn1_var_name
            ],
            outputs=[bn1_output_name],
        )
    ```

  * BH节点输入信息除了 Conv1_output ，还需要BH公式中的其他参数，包括 scale，bias，mean，var：

    * 这些参数和创建 weight 和 bias 一样，通过 Initializer 并实现：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1df2cf09-5c49-4169-9078-0c7d8e6ae6d0)

    * scale，bias，mean，var 的权重信息由随机生成获得：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/353db4ae-63db-42a9-bef1-02dd021c20f7)


* 6、创建 Relu 节点，AveragePool节点：

  * ```python
        ##########################创建一个ReLU节点###########################
        relu1_output_name = "relu1.output"
    
        # 创建ReLU节点，ReLU不需要权重，所以直接make_node就好了
        relu1_node = onnx.helper.make_node(
            name    = "relu1",
            op_type = "Relu", # 要和opset一致
            inputs  = [bn1_output_name],
            outputs = [relu1_output_name],
        )
        
        ##########################创建一个AveragePool节点####################
        avg_pool1_output_name = "avg_pool1.output"
    
        # 创建AvgPool节点，AvgPool不需要权重，所以直接make_node就好了
        avg_pool1_node = onnx.helper.make_node(
            name    = "avg_pool1",
            op_type = "GlobalAveragePool",
            inputs  = [relu1_output_name],
            outputs = [avg_pool1_output_name],
        )
    ```

  * op_type = "Relu", # 要和opset一致

* 7、创建第2个Conv节点，与第1个方法一致：

  * ```python
       ##########################创建第二个conv节点##############################
    
        # 创建conv节点的属性
        conv2_in_ch  = conv1_out_ch
        conv2_out_ch = output_channel
        conv2_kernel = 1
        conv2_pads   = 0
    
        # 创建conv节点的权重信息
        conv2_weight    = np.random.rand(conv2_out_ch, conv2_in_ch, conv2_kernel, conv2_kernel)
        conv2_bias      = np.random.rand(conv2_out_ch)
        
        conv2_weight_name = "conv2d_2.weight"
        conv2_weight_initializer = create_initializer_tensor(
            name         = conv2_weight_name,
            tensor_array = conv2_weight,
            data_type    = onnx.TensorProto.FLOAT)
    
        conv2_bias_name  = "conv2d_2.bias"
        conv2_bias_initializer = create_initializer_tensor(
            name         = conv2_bias_name,
            tensor_array = conv2_bias,
            data_type    = onnx.TensorProto.FLOAT)
    
        # 创建conv节点，注意conv节点的输入有3个: input, w, b
        conv2_node = onnx.helper.make_node(
            name         = "conv2d_2",
            op_type      = "Conv",
            inputs       = [
                avg_pool1_output_name,
                conv2_weight_name,
                conv2_bias_name
            ],
            outputs      = [model_output_name],
            kernel_shape = [conv2_kernel, conv2_kernel],
            pads         = [conv2_pads, conv2_pads, conv2_pads, conv2_pads],
        )
    ```

* 8、创建graph：

  * ```python
        ##########################创建graph##############################
        graph = onnx.helper.make_graph(
            name    = "sample-convnet",
            inputs  = [input],
            outputs = [output],
            nodes   = [
                conv1_node, 
                bn1_node, 
                relu1_node, 
                avg_pool1_node, 
                conv2_node],
            initializer =[
                conv1_weight_initializer, 
                conv1_bias_initializer,
                bn1_scale_initializer, 
                bn1_bias_initializer,
                bn1_mean_initializer, 
                bn1_var_initializer,
                conv2_weight_initializer, 
                conv2_bias_initializer
            ],
        )
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e46a3cce-a77e-4eaa-bd90-23040515c512)


* 9、创建Model，保存验证：

  * ```python
        ##########################创建model##############################
        model = onnx.helper.make_model(graph, producer_name="onnx-sample")
        model.opset_import[0].version = 12
        
        ##########################验证&保存model##############################
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        print("Congratulations!! Succeed in creating {}.onnx".format(graph.name))
        onnx.save(model, "../models/sample-convnet.onnx")
    ```



##### （4）打印模型的其他方法

* Model打印除了用【print(model)】（```不美观```）,还可以用美观一些的方式打印：

  * ```python
    def main(): 
    
        model = onnx.load("../models/sample-linear.onnx")
        onnx.checker.check_model(model)
    
        graph        = model.graph
        nodes        = graph.node
        inputs       = graph.input
        outputs      = graph.output
    
        print("\n**************parse input/output*****************")
        for input in inputs:
            input_shape = []
            for d in input.type.tensor_type.shape.dim:
                if d.dim_value == 0:
                    input_shape.append(None)
                else:
                    input_shape.append(d.dim_value)
            print("Input info: \
                    \n\tname:      {} \
                    \n\tdata Type: {} \
                    \n\tshape:     {}".format(input.name, input.type.tensor_type.elem_type, input_shape))
    
        for output in outputs:
            output_shape = []
            for d in output.type.tensor_type.shape.dim:
                if d.dim_value == 0:
                    output_shape.append(None)
                else:
                    output_shape.append(d.dim_value)
            print("Output info: \
                    \n\tname:      {} \
                    \n\tdata Type: {} \
                    \n\tshape:     {}".format(input.name, output.type.tensor_type.elem_type, input_shape))
    
        print("\n**************parse node************************")
        for node in nodes:
            print("node info: \
                    \n\tname:      {} \
                    \n\top_type:   {} \
                    \n\tinputs:    {} \
                    \n\toutputs:   {}".format(node.name, node.op_type, node.input, node.output))
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/445abe20-a8d6-460b-a592-4184a7d3bc63)


  * 也可以编写打印函数调用，参考【parser.py】和【parser_onnx_cbr.py】

#### 3.4.5 ONNX注册算子的方法

经常会遇到 pytorch 导出 onnx 不成功（不兼容）的时候，需要进行解决

##### （1）转换swin-tiny时候出现的不兼容op的例子

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c8890c54-ba61-4884-8bb4-84322ae23251)


##### （2）当出现导出onnx不成功的时候，我们需要考虑的事情

难易度从低到高：

* 【⭐			】修改opset的版本（opset向上升级调高到兼容的部分）
  * 查看不支持的算子在新的opset中是否被支持（可以参考onnx的文档）
    * [onnx/docs/Operators.md at main · onnx/onnx (github.com)](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
  *  如果不考虑自己搭建plugin的话，也需要看看onnx-trt中这个算子是否被支持
    * 因为onnx是一种图结构表示，并不包含各个算子的实现。除非我们是要在onnx-runtime上测试， 否则我们更看重onnx-trt中这个算子的支持情况
  * 看官方文档
* 【⭐⭐		】替换pytorch中的算子组合
  * 把某些计算替换成onnx可以识别的
* 【⭐⭐⭐	】在pytorch登记onnx中某些算子
  * 有可能onnx中有支持，但没有被登记（Asinh为例，虽然onnx支持，但是没有登记）
* 【⭐⭐⭐⭐】直接修改onnx，创建plugin
  * 使用onnx-surgeon
  * 一般是用在加速某些算子上使用



##### （3）unsupported asinh算子

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3ace20b7-af7b-4012-9f9e-26057a77f036)


* 1、先去寻找官方文档：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/99b07b5c-a31d-48e5-8ac5-c15ce8fc138f)

  * 从onnx支持的算子里面我们可以知道自从opset9开始asinh就已经被支持了
  * 所以可以知道，问题是出现在PyTorch与onnx之间没有建立asinh的映射，需要建立这个映射

* 2、建立映射

  * 寻找pytorch到onnx中的注册算子【官方文档】

    * 这里我的电脑onnx官方注册算子文档与教程不一致：

      * E:\Anaconda\envs\yolov5\Lib\site-packages\torch\onnx
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/22e92266-e4b5-4bce-bf3f-006bf7d81049)


    * 以sub算子为例：

      * ```python
        @_onnx_symbolic("aten::reshape_as")
        @_beartype.beartype
        def sub(g: jit_utils.GraphContext, self, other, alpha=None):
            if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
                other = g.op("Mul", other, alpha)
            return g.op("Sub", self, other)
        ```

      * aten是"a Tensor Library"的缩写，是一个实现张量运算的C++库

      *  aten::xxx

        * c++的一个namespace
        * pytorch的很多算子的底层都是在aten这个命名空间下进行以c++进行实现的

      * onnx_symblic

        * 负责绑定
        * 绑定pytorch中的算子与aten命名空间下的算子的一一对应

  * 【算子注册方法1】那么我们就可以按照类似的方法去自己创建这个联系

    * 创建symbolic符号函数，从而创建一个onnx operator

      * ```python
        # 创建一个asinh算子的symblic，符号函数，用来登记
        # 符号函数内部调用g.op, 为onnx计算图添加Asinh算子
        #   g: 就是graph，计算图
        #   也就是说，在计算图中添加onnx算子
        #   由于我们已经知道Asinh在onnx是有实现的，所以我们只要在g.op调用这个op的名字就好了
        #   symblic的参数需要与Pytorch的asinh接口函数的参数对齐
        #       def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...
        def asinh_symbolic(g, input, *, out=None):
            return g.op("Asinh", input)
        
        # 在这里，将asinh_symbolic这个符号函数，与PyTorch的asinh算子绑定。也就是所谓的“注册算子”
        # asinh是在名为aten的一个c++命名空间下进行实现的
        ```

    * 注册这个onnx operator，并让它与底层的aten中的asinh实现绑定

      * ```python
        register_custom_op_symbolic('aten::asinh', asinh_symbolic, 12)
        ```

      * aten::asinh：底层的C++函数实现

      * asinh_symbolic：symbolic符号函数

      * 12：代表第几个opset开始支持

    * 这里容易混淆的地方：

      * register_op中的第一个参数是PyTorch中的算子名字: aten::asinh
      * g.op中的第一个参数是onnx中的算子名字: Asinh

    * 运行注册好的【sample_asinh_register.py】

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/be8dd616-243f-47e9-bd61-9a929815fb64)

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/48834b81-cedc-435f-af21-90c824de01e9)


  * 【算子注册方法2】

    * ```python
      # 另外一个写法
      #    这个是类似于torch/onnx/symbolic_opset*.py中的写法
      #    通过torch._internal中的registration来注册这个算子，让这个算子可以与底层C++实现的aten::asinh绑定
      #    一般如果这么写的话，其实可以把这个算子直接加入到torch/onnx/symbolic_opset*.py中
      @_onnx_symbolic('aten::asinh')
      def asinh_symbolic(g, input, *, out=None):
          return g.op("Asinh", input)
      ```

##### （4）自创算子（未注册）

* 1、创建一个叫CustomOp的类作为算子：

  * 这个算子可以把0以下的数据的数据变成0（只保存0以上的），并做了一个除法运算

  * ```python
    class CustomOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(x)
            x = x.clamp(min=0) # 做一个截取
            return x / (1 + torch.exp(-x))
    ```

* 2、在Model中导入算子：

  * ```python
    customOp = CustomOp.apply
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            x = customOp(x)
            return x
    ```

* 3、运行结果

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/944b720f-397c-4e64-aa86-f590a6ab3f50)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/599e6a99-09f6-42bf-8621-c8c31c3d83c3)

    * 每一个节点都被跟踪，这样对于网络可视化并不友好，需要简化

##### （5）自创算子（注册）

* 1、创建一个叫CustomOp的类作为算子：

  * 加入注册symbolic

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e820d3c8-5913-4c6e-9f92-917a3192f817)


    * 自己定义一个名称空间：
      * custom_domain::customOp2

  * ```python
    class CustomOp(torch.autograd.Function):
        @staticmethod # 静态的注册方法
        def symbolic(g: torch.Graph, x: torch.Value) -> torch.Value:
            return g.op("custom_domain::customOp2", x)
    
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(x)
            x = x.clamp(min=0)
            return x / (1 + torch.exp(-x))
    ```

* 2、运行结果：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/459958da-d297-47a2-9ee1-5bdadf329804)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/359749bb-fa7f-4a46-b058-e1dfea1fdc53)




##### （6）DeformConv2d

* 1、未注册前【sample_deformable_conv.py】：

  * 输入输出没问题，但是导出是出问题：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2c3fd6a7-218d-4fb2-ba06-c968fd1de9fc)

    * 原因是算子不兼容

* 2、按照之前方法注册算子：

  * ```python
    # 注意
    #   这里需要把args的各个参数的类型都指定
    #   这里还没有实现底层对deform_conv2d的实现
    #   具体dcn的底层实现是在c++完成的，这里会在后面的TensorRT plugin中回到这里继续讲这个案例
    #   这里先知道对于不支持的算子，onnx如何导出即可
    @parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i","i", "i", "i", "none")
    def dcn_symbolic(
            g,
            input,
            weight,
            offset,
            mask,
            bias,
            stride_h, stride_w,
            pad_h, pad_w,
            dil_h, dil_w,
            n_weight_grps,
            n_offset_grps,
            use_mask):
        return g.op("custom::deform_conv2d", input, offset)
    
    register_custom_op_symbolic("torchvision::deform_conv2d", dcn_symbolic, 12)
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d2b96eaf-f073-4417-a4da-eadb18a7a4a2)

* 3、输出

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8d9ed40b-4e26-4974-b391-7e5ec950a746)


#### 3.4.6 ONNX graph surgeon

之前的案例我们都是用```onnx.helper```的API创建修改onnx，但是在TensorRT中也存在修改onnx的工具包：onnx-graph-surgon

##### （1）ONNX graph surgeon安装

* 创建/修改onnx的工具。在TensorRT/tools中可以安装

  * 更加方便的添加/修改onnx节点
  * 更加方便的修改子图
  * 更加方便的替换算子
  * (底层一般是用的onnx.helper，但是给做了一些封装)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b72115dd-2a93-4800-99a1-7492917ec3b5)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5220a3cd-dc48-4a19-a884-8021857f22a0)


* 安装指令：

  * ```python
    python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
    ```

##### （2）Intermediate Representation不同平台中的对比

* 1、onnx ProtoBuf中的IR表示：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6af87d1f-2603-46b0-aba1-278c3e55e4a0)

* 2、onnx graph surgeon中的IR表示：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7df0207e-418f-44ac-8ef7-83151cb1588f)

  * gs帮助我们隐藏了很多信息
  * node的属性以前使用AttributeProto保存， 但是gs中统一用dict来保存

##### （3）Intermediate Representation不同平台创建onnx的对比

* 1、使用原生的onnx.hepler创建onnx：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a89e0bd1-8608-44b6-9291-8560e51ab315)

* 2、使用原生的gs创建onnx
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/67353c0a-b03b-4b91-bacd-eaae5753df62)



##### （4）onnx graph surgeon其他用法

* 1、gs可以自定义一些函数去创建onnx，使整个onnx的创建更加方便（我们完全可以自己创建一些算子在这里使用）：
  * 1.在graph注册调用的函数（类似于onnx中symbolic符号函数来注册算子一样）
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/cfe81da8-3a34-4280-92fe-23dc95e3c11c)

  * 2.设计网络架构
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0f07f3f9-1125-4de0-97c3-f54a71bf1d2a)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f2661195-3d1d-424b-ab99-5c74c9ebff5c)

* 2、gs可以方便我们把整个网络中的一些子图给“挖”出来，以此来分析细节（一般配合polygraphy(*)使用，去寻找量化掉精度严重的子图)
  * 以swin transformer类型的网络架构来看：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bc851aac-916f-4ed6-ad9b-6845045e966c)

  * 可以用gs挖出整个网络中的小部分，例如LayerNorm部分和MHSA部分：
    * LayerNorm部分：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/50c4893f-9735-47ee-be21-cb09d0857b8f)

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2a03fc31-de57-433f-88a7-8e51889a3ba5)

    * MHSA部分：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ca0ae250-9eb9-4a81-bd71-11097104f202)

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b1b41679-fe40-46a6-aae2-4bd1ca3f7839)

* 3、可以使用gs来替换算子或者创建算子（gs中最重要的一个特点）
  * 比如原来的网络：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f1d1ee7e-7039-4230-bfdb-f1cef9157a13)

  * 用gs自己创建一个算子，把想要绑定的算子结合起来另外一个原生算子，提交给TensorRT plugin实现算子加速：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/586223ce-89bc-46b8-8524-969cbb4cf4c9)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6f911c44-85d9-4ccb-a44f-27d04c356208)


##### （5）gs创建conv网络

* ```python
  import onnx_graphsurgeon as gs
  import numpy as np
  import onnx
  
  # onnx_graph_surgeon(gs)中的IR会有以下三种结构
  # Tensor
  #    -- 有两种类型
  #       -- Variable:  主要就是那些不到推理不知道的变量
  #       -- Constant:  不用推理时，而在推理前就知道的变量
  # Node
  #    -- 跟onnx中的NodeProto差不多
  # Graph
  #    -- 跟onnx中的GraphProto差不多
  
  def main() -> None:
      input = gs.Variable(
              name  = "input0",
              dtype = np.float32,
              shape = (1, 3, 224, 224))
  
      weight = gs.Constant(
              name  = "conv1.weight",
              values = np.random.randn(5, 3, 3, 3))
  
      bias   = gs.Constant(
              name  = "conv1.bias",
              values = np.random.randn(5))
      
      output = gs.Variable(
              name  = "output0",
              dtype = np.float32,
              shape = (1, 5, 224, 224))
  
      node = gs.Node(
              op      = "Conv",
              inputs  = [input, weight, bias],
              outputs = [output],
              attrs   = {"pads":[1, 1, 1, 1]})
  
      graph = gs.Graph(
              nodes   = [node],
              inputs  = [input],
              outputs = [output])
  
      model = gs.export_onnx(graph)
  
      onnx.save(model, "../models/sample-conv.onnx")
  
  
  
  # 使用onnx.helper创建一个最基本的ConvNet
  #         input (ch=3, h=64, w=64)
  #           |
  #          Conv (in_ch=3, out_ch=32, kernel=3, pads=1)
  #           |
  #         output (ch=5, h=64, w=64)
  
  if __name__ == "__main__":
      main()
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9e3877ef-5e8d-4516-9c47-e3ec7c36760d)

##### （6）gs自定义一些函数网络

* ```python
  import onnx_graphsurgeon as gs
  import numpy as np
  import onnx
  
  #####################在graph注册调用的函数########################
  @gs.Graph.register()
  def add(self, a, b):
      return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])
  
  @gs.Graph.register()
  def mul(self, a, b):
      return self.layer(op="Mul", inputs=[a, b], outputs=["mul_out_gs"])
  
  # 做矩阵乘法
  @gs.Graph.register()
  def gemm(self, a, b, trans_a=False, trans_b=False):
      # attrs是属性，用字典类型保存，用于做矩阵乘法时表示哪一个矩阵做转秩
      attrs = {"transA": int(trans_a), "transB": int(trans_b)}
      return self.layer(op="Gemm", inputs=[a, b], outputs=["gemm_out_gs"], attrs=attrs) 
  
  @gs.Graph.register()
  def relu(self, a):
      return self.layer(op="Relu", inputs=[a], outputs=["act_out_gs"])
  
  #####################通过注册的函数进行创建网络########################
  #          input (64, 64)
  #            |
  #           gemm (constant tensor A(64, 32))
  #            |
  #           add  (constant tensor B(64, 32))
  #            |
  #           relu
  #            |
  #           mul  (constant tensor C(64, 32))
  #            |
  #           add  (constant tensor D(64, 32))
  
  # 初始化网络结构
  graph    = gs.Graph(opset=12)
  
  # 初始化网络需要用的参数
  consA    = gs.Constant(name="consA", values=np.random.randn(64, 32))
  consB    = gs.Constant(name="consB", values=np.random.randn(64, 32))
  consC    = gs.Constant(name="consC", values=np.random.randn(64, 32))
  consD    = gs.Constant(name="consD", values=np.random.randn(64, 32))
  input0   = gs.Variable(name="input0", dtype=np.float32, shape=(64, 64))
  
  # 设计网络架构
  gemm0    = graph.gemm(input0, consA, trans_b=True)
  relu0    = graph.relu(*graph.add(*gemm0, consB))
  mul0     = graph.mul(*relu0, consC)
  output0  = graph.add(*mul0, consD)
  
  # 设置网络的输入输出
  graph.inputs = [input0]
  graph.outputs = output0
  
  for out in graph.outputs:
      out.dtype = np.float32
  
  # 保存模型
  onnx.save(gs.export_onnx(graph), "../models/sample-complicated-graph1.onnx")
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/211d7537-9e4d-4956-99a3-41de5837f4b6)


##### （7）gs挖出整个网络中的小部分

* 1、针对xxx需要截取LayerNormalization部分和selfattention（MHSA）部分截取出来：

  * LayerNormalization部分：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e7748069-1e11-4225-8bbc-72490600a4ee)

  * selfattention（MHSA）部分：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bbc6f6d2-a945-4b99-9f0d-0da1bf41f3d2)


* 2、网络结构已经从3.6models中的opset12的onnx找到。已经更新到了3.5models中。

* 3、打印需要提取部分的输入输出内容：

  * LayerNorm部分的输入输出，首先在onnx图中找到它们的输入输出的name：

    * 输入口【374】

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/83cc54dc-bead-4f40-9a5d-d2c3b5d3b0b9)


    * 输出口【383】

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4f13f02a-66b6-4e2c-8ed9-caed4fc41a17)


    * 打印输入输出

      * ```python
        import onnx_graphsurgeon as gs
        import numpy as np
        import onnx
        
        def load_model(model : onnx.ModelProto):
            graph = gs.import_onnx(model)
            print(graph.inputs)
            print(graph.outputs)
        
        def main() -> None:
            model = onnx.load("../models/swin-tiny.onnx")
        
            graph = gs.import_onnx(model)
            tensors = graph.tensors()
        
            # LayerNorm部分
            print(tensors["374"]) # LN的input1: 1 x 3136 x 128
            print(tensors["383"]) # LN的输出:   1 x 3136 x 128
        
        
        # 我们想把swin中LayerNorm这一部分单独拿出来
        if __name__ == "__main__":
            main()
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/307c7431-9720-4b9e-adb6-3f780cfb8b2e)


    * 提取输入输出范围内的子部分：

      * ```python
        import onnx_graphsurgeon as gs
        import numpy as np
        import onnx
        
        def load_model(model : onnx.ModelProto):
            graph = gs.import_onnx(model)
            print(graph.inputs)
            print(graph.outputs)
        
        def main() -> None:
            model = onnx.load("../models/swin-tiny.onnx")
        
            graph = gs.import_onnx(model)
            tensors = graph.tensors()
        
            # LayerNorm部分
            print(tensors["374"]) # LN的input1: 1(Batch) x 3136(L=HxW) x 128(Size)
            print(tensors["383"]) # LN的输出:   1 x 3136 x 128
        
            # 重置graph中的输入和输出
            graph.inputs = [
                tensors["374"].to_variable(dtype=np.float32, shape=(1, 3136, 128))]
            graph.outputs = [
                tensors["383"].to_variable(dtype=np.float32, shape=(1, 3136, 128))]
            
            # 清楚输入输出之外的部分
            graph.cleanup()
            onnx.save(gs.export_onnx(graph), "../models/swin-subgraph-LN.onnx")
        
        
        # 我们想把swin中LayerNorm这一部分单独拿出来
        if __name__ == "__main__":
            main()
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3ee76cc8-4bf3-4874-bd68-8d944077a3d8)


  * MHSA部分的输入输出，首先在onnx图中找到它们的输入输出的name：

    * 输入口【457】

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/143801ef-6646-4bd3-9815-0f26d3ee9567)


    * 输出口【512】

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/48a21918-c021-4cfa-bfcc-528ff6bc279c)


    * 由于矩阵乘法中存在权重信息，因此也需要找到：

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c3a2c7a8-e628-4e4f-88e2-484bb2a28f09)

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e7226650-b38d-4e00-bc58-00398c86b513)


    * 打印：

      * ```python
        import onnx_graphsurgeon as gs
        import numpy as np
        import onnx
        
        def load_model(model : onnx.ModelProto):
            graph = gs.import_onnx(model)
            print(graph.inputs)
            print(graph.outputs)
        
        def main() -> None:
            model = onnx.load("../models/swin-tiny.onnx")
            # MHSA部分
            graph = gs.import_onnx(model)
            tensors = graph.tensors()
            print(tensors["457"]) # MHSA输入matmul:       64 x 49 x 128
            print(tensors["5509"])  # MHSA输入matmul的权重: 128 x 384
            print(tensors["5518"])  # MHSA输出matmul的权重: 128 x 128
            print(tensors["512"]) # MHSA输出:             64 x 49 x 128
        
        # 我们想把swin中LayerNorm这一部分单独拿出来
        if __name__ == "__main__":
            main()
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/89bc567e-c443-46fb-a929-775f330fba15)


    * 提取输入输出范围内的子部分：

      * ```python
        import onnx_graphsurgeon as gs
        import numpy as np
        import onnx
        
        def load_model(model : onnx.ModelProto):
            graph = gs.import_onnx(model)
            print(graph.inputs)
            print(graph.outputs)
        
        def main() -> None:
            model = onnx.load("../models/swin-tiny.onnx")
            # MHSA部分
            graph = gs.import_onnx(model)
            tensors = graph.tensors()
            print(tensors["457"]) # MHSA输入matmul:       64 x 49 x 128
            print(tensors["5509"])  # MHSA输入matmul的权重: 128 x 384
            print(tensors["5518"])  # MHSA输出matmul的权重: 128 x 128
            print(tensors["512"]) # MHSA输出:             64 x 49 x 128
        
            # 重置graph中的输入和输出
            graph.inputs = [
                tensors["457"].to_variable(dtype=np.float32, shape=(1, 3136, 128))]
            graph.outputs = [
                tensors["512"].to_variable(dtype=np.float32, shape=(1, 3136, 128))]
            
            # 清楚输入输出之外的部分
            graph.cleanup()
            onnx.save(gs.export_onnx(graph), "../models/swin-subgraph-MSHA.onnx")
        
        # 我们想把swin中MHSA这一部分单独拿出来
        if __name__ == "__main__":
            main()
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/80e9f46f-510a-4209-bddc-d75ecea6492f)


##### （8）gs替换算子或者创建算子（clip）

* 1、将网络内部的算子替换：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5a997d7d-89da-4496-96e4-100cbeabcb3d)


* 2、替换前的minmax：

  * ```python
    import onnx_graphsurgeon as gs
    import numpy as np
    import onnx
    import onnxruntime
    import torch
    
    #####################在graph注册调用的函数########################
    @gs.Graph.register()
    def min(self, *args):
        return self.layer(op="Min", inputs=args, outputs=["min_output"])
    
    @gs.Graph.register()
    def max(self, *args):
        return self.layer(op="Max", inputs=args, outputs=["max_output"])
    
    #####################通过注册的函数进行创建网络########################
    #          input (5, 5)
    #            |
    #         identity 
    #            |
    #           min  
    #            |
    #           max
    #            |
    #         identity  
    #            |
    #          output (5, 5)
    def create_onnx_graph():
        # 初始化网络的opset
        graph    = gs.Graph(opset=12)
    
        # 初始化网络需要用的参数
        min_val  = np.array(0, dtype=np.float32)
        max_val  = np.array(1, dtype=np.float32)
        input0   = gs.Variable(name="input0", dtype=np.float32, shape=(5, 5))
    
        # 设计网络架构
        identity0 = graph.identity(input0)
        min0      = graph.min(*identity0, max_val)
        max0      = graph.max(*min0, min_val)
        output0   = graph.identity(*max0)
    
        # 设置网络的输入输出
        graph.inputs = [input0]
        graph.outputs = output0
    
        # 设置网络的输出的数据类型
        for out in graph.outputs:
            out.dtype = np.float32
    
        # 保存模型
        onnx.save(gs.export_onnx(graph), "../models/sample-minmax.onnx")
    
    #####################验证模型##########################################
    def validate_onnx_graph(input, path):
        sess   = onnxruntime.InferenceSession(path)
        output = sess.run(None, {'input0': input.numpy()})
    
        print("input is \n", input)
        print("output is \n", output)
    
    
        
    
    def main() -> None:
        input  = torch.Tensor(5, 5).uniform_(-1, 1)
    
        # 创建一个minmax的网络
        create_onnx_graph()
    
        # 通过onnxruntime确认导出onnx是否正确生成
        print("\nBefore modification:")
        validate_onnx_graph(input, "../models/sample-minmax.onnx")
    
    if __name__ == "__main__":
        main()
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/75ee53a8-b99e-4052-bce7-196225507547)


* 3、替换后的clip：

  * ```python
    import onnx_graphsurgeon as gs
    import numpy as np
    import onnx
    import onnxruntime
    import torch
    
    #####################在graph注册调用的函数########################
    @gs.Graph.register()
    def identity(self, a):
        return self.layer(op="Identity", inputs=[a], outputs=["identity_output"])
    
    @gs.Graph.register()
    def clip(self, inputs, outputs):
        return self.layer(op="Clip", inputs=inputs, outputs=outputs)
    
    #####################通过注册的clip算子替换网络节点####################
    #          input (5, 5)
    #            |
    #         identity 
    #            |
    #           clip
    #            |
    #         identity  
    #            |
    #          output (5, 5)
    def change_onnx_graph():
        graph = gs.import_onnx(onnx.load_model('../models/sample-minmax.onnx'))
        tensors = graph.tensors()
    
        inputs = [tensors["identity_output_0"], 
                  tensors["onnx_graphsurgeon_constant_5"],
                  tensors["onnx_graphsurgeon_constant_2"]]
    
        outputs = [tensors["max_output_6"]]
        
        # 因为要替换子网，所以需要把子网和周围的所有节点都断开联系
        for item in inputs:
            # print(item.outputs)
            item.outputs.clear()
    
        for item in outputs:
            # print(item.inputs)
            item.inputs.clear()
    
        # 通过注册的clip，重新把断开的联系链接起来
        graph.clip(inputs, outputs)
    
        # 删除所有额外的节点
        graph.cleanup()
    
        onnx.save(gs.export_onnx(graph), "../models/sample-minmax-to-clip.onnx")
    
    #####################验证模型##########################################
    def validate_onnx_graph(input, path):
        sess   = onnxruntime.InferenceSession(path)
        output = sess.run(None, {'input0': input.numpy()})
    
        print("input is \n", input)
        print("output is \n", output)
    
    def main() -> None:
        input  = torch.Tensor(5, 5).uniform_(-1, 1)
    
        # 创建一个minmax的网络
        create_onnx_graph()
    
        # 通过onnxruntime确认导出onnx是否正确生成
        print("\nBefore modification:")
        validate_onnx_graph(input, "../models/sample-minmax.onnx")
    
        # 将minmax网络修改成clip网络
        change_onnx_graph()
    
        # 确认网络修改的结构是否正确
        print("\nAfter modification:")
        validate_onnx_graph(input, "../models/sample-minmax-to-clip.onnx")
    
    
    if __name__ == "__main__":
        main()
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/908f64ca-da24-4c59-be54-bfec866bf48a)




##### （9）gs替换算子或者创建算子（LN）

* 1、常见的Transformer子块：

  * ```python
    import onnx_graphsurgeon as gs
    import numpy as np
    import onnx
    import onnxsim
    import onnxruntime
    import torch
    import torch.nn as nn
    
    
    #####################在graph注册调用的函数########################
    @gs.Graph.register()
    def identity(self, inputs, outputs):
        return self.layer(op="Identity", inputs=inputs, outputs=outputs)
    
    @gs.Graph.register()
    def layerNorm(self, inputs, outputs, axis, epsilon):
        attrs = {'axis': np.int64(axis), 'epsilon': np.float(epsilon)}
        return self.layer(op="LayerNormalization", inputs=inputs, outputs=outputs, attrs=attrs)
    
    @gs.Graph.register()
    def layerNorm_default(self, inputs, outputs):
        return self.layer(op="LayerNormalization", inputs=inputs, outputs=outputs)
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.norm  = nn.LayerNorm(3)
            self.act   = nn.ReLU()
    
        # 常见的Transformer子块
        def forward(self, x):
            _, _, H, W = x.shape
            L = H * W
            x = self.conv1(x)
            x = x.view(x.shape[0], x.shape[1], L).permute(0, 2, 1) # 把BCHW转成BLC
            x = self.norm(x)
            x = self.act(x)
            return x
    
    def export_onnx_graph():
        input  = torch.Tensor(1, 3, 5, 5).uniform_(-1, 1)
        model  = Model()
        model.eval()
    
        file   = "../models/sample-ln-before.onnx"
        torch.onnx.export(
                model         = model,
                args          = (input,),
                f             = file,
                input_names   = ["input0"],
                output_names  = ["output0"],
                opset_version = 12)
    
        print("\nFinished export {}".format(file))
    
        model_onnx = onnx.load(file)
        onnx.checker.check_model(model_onnx)
    
        print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, "assert check failed"
        onnx.save(model_onnx, file)
    
    def main() -> None:
        input  = torch.Tensor(1, 3, 5, 5).uniform_(-1, 1)
        
        ##从pytorch导出onnx(这里为了实验，将不支持LayerNorm的opset12为例导出)
        export_onnx_graph()
    
    if __name__ == "__main__":
        main()
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0768e3b4-6000-4250-a82f-378ea731990a)


* 2、将网络内部的算子替换成LayerNorm：

  * ```python
    import onnx_graphsurgeon as gs
    import numpy as np
    import onnx
    import onnxsim
    import onnxruntime
    import torch
    import torch.nn as nn
    
    
    #####################在graph注册调用的函数########################
    @gs.Graph.register()
    def identity(self, inputs, outputs):
        return self.layer(op="Identity", inputs=inputs, outputs=outputs)
    
    @gs.Graph.register()
    def layerNorm(self, inputs, outputs, axis, epsilon):
        attrs = {'axis': np.int64(axis), 'epsilon': np.float(epsilon)}
        return self.layer(op="LayerNormalization", inputs=inputs, outputs=outputs, attrs=attrs)
    
    @gs.Graph.register()
    def layerNorm_default(self, inputs, outputs):
        return self.layer(op="LayerNormalization", inputs=inputs, outputs=outputs)
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
            self.norm  = nn.LayerNorm(3)
            self.act   = nn.ReLU()
    
        # 常见的Transformer子块
        def forward(self, x):
            _, _, H, W = x.shape
            L = H * W
            x = self.conv1(x)
            x = x.view(x.shape[0], x.shape[1], L).permute(0, 2, 1) # 把BCHW转成BLC
            x = self.norm(x)
            x = self.act(x)
            return x
    
    def export_onnx_graph():
        input  = torch.Tensor(1, 3, 5, 5).uniform_(-1, 1)
        model  = Model()
        model.eval()
    
        file   = "../models/sample-ln-before.onnx"
        torch.onnx.export(
                model         = model,
                args          = (input,),
                f             = file,
                input_names   = ["input0"],
                output_names  = ["output0"],
                opset_version = 12)
    
        print("\nFinished export {}".format(file))
    
        model_onnx = onnx.load(file)
        onnx.checker.check_model(model_onnx)
    
        print(f"Simplifying with onnx-simplifier {onnxsim.__version__}...")
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, "assert check failed"
        onnx.save(model_onnx, file)
    
    
    #####################通过注册的LN算子替换网络节点####################
    #          input (5, 5)
    #            |
    #           conv
    #            |
    #          reshape
    #            |
    #         layerNorm
    #            |
    #           relu
    #            |
    #          output (5, 5)
    
    def change_onnx_graph():
        graph = gs.import_onnx(onnx.load_model('../models/sample-ln-before.onnx'))
        tensors = graph.tensors()
    
        norm_scale = gs.Constant(name="norm.weight", values=np.ones(shape=[3], dtype=np.float32)) # weight全部归一
        norm_bias  = gs.Constant(name="norm.bias", values=np.zeros(shape=[3], dtype=np.float32)) # bias全部归零
    
        inputs  = [tensors["/Transpose_output_0"]]
        outputs = [tensors["/norm/Div_output_0"]]
        
        # 因为要替换子网，所以需要把子网和周围的所有节点都断开联系
        for item in inputs:
            item.outputs.clear()
    
        for item in outputs:
            item.inputs.clear()
    
        # 为了迎合onnx中operator中的设计，这里把scale和bias给加上
        inputs = [tensors["/Transpose_output_0"],
                  norm_scale,
                  norm_bias]
        
        # 这个onnx中的epsilon，我们给加上。当然，我们也可以选择默认的值
        epsilon = [tensors["/norm/Constant_1_output_0"]]
        print(type(epsilon[0].values))
    
        # 通过注册的LayerNorm，重新把断开的联系链接起来
        graph.layerNorm(inputs, outputs, axis=-1, epsilon=epsilon[0].values)
        # graph.identity(inputs, outputs)
        # graph.layerNorm_default(inputs, outputs)
    
        # 删除所有额外的节点
        graph.cleanup()
    
        onnx.save(gs.export_onnx(graph), "../models/sample-ln-after.onnx")
    
    #####################验证模型##########################################
    def validate_onnx_graph(input, origin_path, modified_path):
        sess_origin   = onnxruntime.InferenceSession(origin_path)
        output_origin = sess_origin.run(None, {'input0': input.numpy()})
    
        sess_modify   = onnxruntime.InferenceSession(modified_path)
        output_modify = sess_modify.run(None, {'input0': input.numpy()})
    
        print("input is \n", input)
        print("output_before is \n", output_origin)
        print("output_after is \n", output_modify)
    
    
    def main() -> None:
        input  = torch.Tensor(1, 3, 5, 5).uniform_(-1, 1)
        
        ##从pytorch导出onnx(这里为了实验，将不支持LayerNorm的opset12为例导出)
        export_onnx_graph()
    
        ##手动修改LayerNorm
        change_onnx_graph()
    
        ##验证修改的onnx
        validate_onnx_graph(
                input, 
                "../models/sample-ln-before.onnx", 
                "../models/sample-ln-after.onnx")
    
    if __name__ == "__main__":
        main()
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3d73692b-18d8-419b-aa88-7e3b25c3a30f)


#### 3.4.7 快速分析开源代码并导出onnx

以Swin Transformer为例学习快速导出onnx并分析onnx的方法，由于搭配环境和修改代码消耗时间过多，而导出onnx并不是重点，重点是onnx导出为TensorRT并查看性能。因此本部分内容在后续需要时看PPT和视频3.8即可

### 3.5 初步使用TensorRT

#### 3.5.1 推理引擎生成

![image-20240514164108751](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240514164108751.png)

中间环形部分就是推理引擎，那么如何生成推理引擎呢，有两个方法：

##### （1）TensorRT API

使用TensorRT API来实现创建推理引擎（C++/python）

后续章节会学习

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b37a865c-d9d4-4c31-8f50-d4798d3d5383)


##### （2）使用命令行（trtexec）

使用命令行创建推理引擎（trtexec）

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/eba7a2fa-546c-4c34-b581-65045f30b89a)


![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/990b4a03-3ff6-4df3-a5ae-6c78032b7da5)


![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c3b128ed-f288-441a-8dac-a4109f9cc3cd)


#### 3.5.2 trtexec

学习使用trtexec，以及通过trtexec的日志理解TensorRT内部所做的优化

是TensorRT中的一个非常好用的快速生成trt推理引擎的工具。便于我们不使用API快速查看推理引擎的加速效果，以及TensorRT优化后的模型架构。

##### （1）指令手册

* 编写命令行的时候可以查询指令手册：
  * 网址：
    * [Developer Guide :: NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-benchmark)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f4589121-5e03-4ec2-b531-754c389debb1)


  * 视频里介绍了常用指令的简介，详情查阅视频3.9

##### （2）代码详情

* build.sh：

  * 程序源码

    * ```python
      #!/bin/bash
      # how to use:
      #   bash tools/build.sh ${input.onnx} ${tag}
      
      # 找文件目录
      IFS=. file=(${1})
      IFS=/ file=(${file})
      IFS=
      PREFIX=${file[1]}
      
      
      if [[ ${2} != "" ]]
      then
              PREFIX=${PREFIX}-${2}
      fi
      
      MODE="build"
      ONNX_PATH="models"
      BUILD_PATH="build"
      ENGINE_PATH=$BUILD_PATH/engines
      LOG_PATH=${BUILD_PATH}"/log/"${PREFIX}"/"${MODE}
      
      mkdir -p ${ENGINE_PATH}
      mkdir -p $LOG_PATH
      
      # trtexec
      # warmup：# 有些kernel执行的比较慢，所以需要使用warmup对齐时间，200是200ms
      # iterations：指做了多上次推理，50就是50次推理
      trtexec --onnx=${1} \
              --memPoolSize=workspace:2048 \
              --saveEngine=${ENGINE_PATH}/${PREFIX}.engine \
              --profilingVerbosity=detailed \
              --dumpOutput \
              --dumpProfile \
              --dumpLayerInfo \
              --exportOutput=${LOG_PATH}/build_output.log\
              --exportProfile=${LOG_PATH}/build_profile.log \
              --exportLayerInfo=${LOG_PATH}/build_layer_info.log \
              --warmUp=200 \
              --iterations=50 \
      
              > ${LOG_PATH}/build.log
      ```

  * 启动指令

    * ```python
      bash tools/build.sh  models/sample-cbr.onnx fp32
      ```

    * ![image-20240514173003852](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240514173003852.png)

  * 导出文件

    * build_layer_info.log
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f865af14-4ed5-43f7-9e98-db230da0d103)

    * build_output.log
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/936ad5dd-9228-42cd-84f4-3a3de92f6131)

    * build_profile.log
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/da41de14-398a-4c9a-a19d-f7d022464178)

      * 这里需要注意：原来网络中Conv和Relu算子之间是分开的
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2deaffce-773b-41b4-87dd-50e423df2d47)

      * 经过trtexec后，他们之间会被融合：
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7c5c79ca-ce5f-4c29-8136-8f016322eb1c)


  * infer.sh：

    * 程序源码

      * ```python
        #!/bin/bash
        # how to use:
        #   bash tools/infer.sh ${input.engine}
        
        IFS=. file=(${1})
        IFS=/ file=(${file})
        IFS=
        PREFIX=${file[2]}
        
        
        if [[ ${2} != "" ]]
        then
                PREFIX=${PREFIX}-${2}
        fi
        
        MODE="infer"
        ONNX_PATH="models"
        BUILD_PATH="build"
        ENGINE_PATH=$BUILD_PATH/engines
        LOG_PATH=${BUILD_PATH}"/log/"${PREFIX}"/"${MODE}
        
        mkdir -p ${ENGINE_PATH}
        mkdir -p $LOG_PATH
        
        trtexec --loadEngine=${ENGINE_PATH}/${PREFIX}.engine \
                --dumpOutput \
                --dumpProfile \
                --dumpLayerInfo \
                --exportOutput=${LOG_PATH}/infer_output.log\
                --exportProfile=${LOG_PATH}/infer_profile.log \
                --exportLayerInfo=${LOG_PATH}/infer_layer_info.log \
                --warmUp=200 \
                --iterations=50 \
                > ${LOG_PATH}/infer.log
        ```

    * 启动指令

      * ```python
        bash tools/infer.sh build/engines/sample-cbr-fp32.engine
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/370fc70e-039e-4422-9966-d3f822dfc775)


    * 导出文件

      * infer_layer_info.log
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4ffbc7c6-f05c-4ea0-9344-7b6bf217644e)

      * infer_output.log
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5c841451-b18c-4436-8ae6-7cf68e8a6cbb)

      * infer_profile.log
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ca15e94e-4143-42f7-9b9d-1c1748901efd)


* profile.sh：

  * 程序源码

    * ```python
      #!/bin/bash
      # how to use:
      #   bash tools/profile.sh ${input.engine} 
      
      IFS=. file=(${1})
      IFS=/ file=(${file})
      IFS=
      PREFIX=${file[2]}
      
      
      if [[ ${2} != "" ]]
      then
              PREFIX=${PREFIX}-${2}
      fi
      
      MODE="profile"
      ONNX_PATH="models"
      BUILD_PATH="build"
      ENGINE_PATH=$BUILD_PATH/engines
      LOG_PATH=${BUILD_PATH}"/log/"${PREFIX}"/"${MODE}
      
      mkdir -p ${ENGINE_PATH}
      mkdir -p $LOG_PATH
      
      # nsys是 Nsight system 的一个软件包
      # trtexec可以配合nsys一起使用进行profile分析
      nsys profile \
              --output=${LOG_PATH}/${PREFIX} \
              --force-overwrite true \
              trtexec --loadEngine=${ENGINE_PATH}/${PREFIX}.engine \
                      --warmUp=0 \
                      --duration=0 \
                      --iterations=20 \
                      --noDataTransfers \
          > ${LOG_PATH}/profile.log
      
      
      ```

  * 启动指令

    * ```python
      bash tools/profile.sh build/engines/sample-cbr-fp32.engine 
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1743ec38-ac4c-48d0-a2af-7e977c7fa577)


  * 输出文件

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ae60b6b5-c5c5-4485-9417-db13e71c6f6f)

    * 其中【sample-cbr-fp32.nsys-rep】需要用Nsight system打开，直接将该文件拖入Nsight中即可
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/819f5f57-d2d5-49a6-8702-9eff8fa37208)

#### 3.5.3 trtexec log分析

trtexec log提供了TensorRT推理优化过程的记录，其中有几点需要注意一下：

##### （1）指定log保存路径

* 指令集：

  * ```python
    > ${LOG_PATH}/build.log
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/88c576d5-4a43-437b-bef8-d3c71119afe4)


##### （2）简易log信息

输出日志：

* 包括输入文件信息、精度、稀疏化等：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8b6b455e-d6e4-4ffb-a5b2-511b3967d5a3)


* 系统选项、推理选项：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/31cd9def-2fec-463f-ab7a-ac277f979160)


* 报告选项、设备信息：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3a15c0e4-b425-46a2-935d-853be0b05447)


* **模型推理创建内容**：

  * 初始化、parsing过程

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e9266d14-0deb-4c96-afa4-76d38051e37e)


    * 其中parsing部分是使用自带的 onnx parser 把 onnx 模型 parse成 TRT 能够识别的 IR；

    * 这个log里面的parsing部分比较简洁，是因为系统自己隐藏，包括层融合、每一层的优化、核优化等，如果要打印，需要在build文件中加入指令：

      * ```python
        --verbose \
        ```

  * 检测到1个输入3个输出：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9b3067a2-2899-4172-8824-50692f845476)


  * 推理部分，load一个engine进行推理：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3f6261ed-087a-40b2-a6a2-49be00a1e341)


  * 最后得到推理出来的东西：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0ac3eb5c-a829-4827-9119-40ceaac8966a)


##### （3）扩展log信息

* 通过上述指令，扩展【V】部分的详细信息：

  * ```python
    --verbose \
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/18818f76-462c-4c29-ab74-127a15a9b19b)


* 打印出来TensorRT中存在的插件

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/87c72e47-e5ba-4f32-9cb2-a89ac435ddf0)


* 注册plugin插件：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/36c81eb4-d633-4bc3-a1b3-1e2438f698c7)


* 分析onnx parser：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b596d257-e29b-4440-b176-fa2cc7c0dec1)

  * 导入输入：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6747f044-f939-4f58-9355-3e23409c14e1)

  * 导入权重信息
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0b11dbe7-a846-4fc0-85c9-bb28481c35b4)

  * 获得输入信息
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/570344a5-7881-4644-b060-c008bc9d51b1)

  * 注册node：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/11615a46-483f-45bb-93c8-7d6d77e99adf)

  * 注册输出：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0052239b-dc4c-4703-baee-d5e1a82679fa)

  * parse RELU
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d0bed848-c9e8-4343-91e2-4eb45cbc63c9)

  * 图优化，包括删除和应用某些模块：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9769698a-b21e-44be-8d21-908a527082d3)

  * kernel优化准备工作：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5ea2bb6b-9664-4ec7-b08e-1432f51606c9)

  * 正式优化：
    * 多次reformat，选取最快的tactic，作为优化策略
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/dd30b52a-55fd-4299-9d56-b95a90c4b1a2)

    * 针对conv和relu进行优化
      * 每个kernel对应不同的layer，layer的不同体现在模型内部大小：
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6e62b7a1-5c25-4623-8671-50cee570e0c7)

      * TensorRT利用多个策略进行layer分析，选择最快的策略：
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/98cfab17-6eea-4ec0-ac45-552f4d3dbaa0)

      * 为了消除偶然性，重复多次优化找最快策略：
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/84f10eeb-6b92-43fd-835a-d00a577e8c2a)

        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/dce50187-0626-4eb2-8e75-080dab91cb5f)

  * 打印信息并汇总：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bb586b87-41ca-4e69-b208-00439fbf9f0a)

  * 推理部分：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/69c14664-061f-4e63-bbb6-4809f77b5e67)




##### （4）量化

以FP16精度量化为例：

* ```python
  --fp16 \
  ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0b140fcb-3043-4a8c-b288-a2c2dc623280)

* ```python
  bash tools/build.sh  models/sample-cbr.onnx fp16
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/96e68acd-aa8a-4614-a56e-cdd0a091fb41)


生成日志：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/eb797acf-e6ca-405f-87f0-e0a5f3dcdd94)



