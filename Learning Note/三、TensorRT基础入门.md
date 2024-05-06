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
  * ![image-20240505163502896](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240505163502896.png)

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
    * ![image-20240506153545906](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506153545906.png)

* 3、创建节点连接的网络：

  * ```python
        # 创建NodeProto
        graph = helper.make_graph([mul, add], 'sample-linear', [a, x, b], [y])
    ```

  * ![image-20240506154002768](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506154002768.png)

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

  * ![image-20240506155747416](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506155747416.png)

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

  * ![image-20240506162138655](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506162138655.png)

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

  * ![image-20240506162455321](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506162455321.png)

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

    * ![image-20240506163333422](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506163333422.png)
      * Input（上一个）；
      * Conv1_weight；
      * Conv1_bias。

  * 其中Conv1_weight 和 Conv1_bais 需要用 Initializer 创建各自的权重：

    * ![image-20240506163239058](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506163239058.png)

  * 输出除了outputs，还要分配 kernel_shape 和 pads：

    * ![image-20240506163353808](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506163353808.png)

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
      * ![image-20240506164702694](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506164702694.png)
    * scale，bias，mean，var 的权重信息由随机生成获得：
      * ![image-20240506164742363](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506164742363.png)

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

  * ![image-20240506170204666](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506170204666.png)

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

  * ![image-20240506170612902](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240506170612902.png)

  * 也可以编写打印函数调用，参考【parser.py】和【parser_onnx_cbr.py】

#### 3.4.5 ONNX注册算子的方法

##### （1）xxx


### 3.5 初步使用TensorRT
