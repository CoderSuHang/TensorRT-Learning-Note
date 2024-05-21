## 四、TensorRT基础入门

### 4.1 模型部署的基础知识

#### 4.1.1 FLOPS 与 TOPS

理解 FLOPS 和 TOPS 是什么，CPU/GPU 中的计算 FLOPS/TOPS 的公式，以及CUDA Core 和 Tensor Core 的区别

##### （1）相关概念

* 1、FLOPS
  * 指的是一秒钟可以处理的浮动小数点运算次数
  * 是衡量计算机硬件性能、计算能力的一个单位
  * 常见的FLOPS：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2b972a04-baff-49a0-beb9-b269279abfbc)




* 2、TOPS
  * 指的是一秒钟可以处理的整型运算次数
  * 是衡量计算机硬件性能、计算能力的一个单位
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a83ca1b7-2b9f-4756-b05f-9ec072b1dbdf)




* 3、FLOPs
  * 是衡量模型大小的一个指标，大家在CVPR的paper或者Github里经常能够看到的就是这个信息



##### （2）FLOPS 在 CPU 中是如何计算的

* 公式：
  * FLOPS = 频率 * core数量 * 每个时钟周期可以处理的 FLOPS
    * 频率：时钟频率
    * Core：硬件核数量
    * 每个时钟周期可以处理的 FLOPS
* 示例：
  * （Intel i7 Haswell架构）8核，频率3.0GHz：
    * FLOPS在双精度的时候：
      * 3.0 * 10^9Hz * 8 core * 16 FLOPS/clk = 0.38 TFLOPS
    * FLOPS在单精度的时候：
      * 3.0 * 10^9Hz * 8 core * 32 FLOPS/clk = 0.76 TFLOPS
    * 计算细节：
      * [1] 在该芯片内部有2个FMA，以及支持AVX-256指令集：
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/068b07a3-452e-47f9-864d-6054d98dff21)

      * [2] FMA是乘加运算混合的一种方法
        * 没有FMA，乘法加法分开算
          * 计算D = A * B + C需要两个时钟周期
          * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2616d6d4-04ca-40a3-bf13-916302722829)

        * 有FMA，乘法加法一起算
          * 计算D = A * B + C需要一个时钟周期
          * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3dfba1d5-d511-470a-b5f3-6aaf90490040)

      * [3] AVX-256 指令集中一个Double指令能存2个Float指令，所以在SIMD操作时，一个时钟周期就能操作8个FP32的计算
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8ff99e17-45ca-4617-83a6-a048ec45b409)

      * [4] 回归公式：
        * FLOPS在双精度的时候：
          * 3.0 * 10^9Hz * 8 core * 16 FLOPS/clk = 0.38 TFLOPS
            * 16 FLOPS/clk = 2 FMA * 4个 FP64 的 SIMD 运算 * 2乘加融合
              * 2 FMA：
                * 一个时钟周期等于2个浮点运算
              * 4个 FP64 的 SIMD 运算：
                * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c4a20d3f-f1cb-492b-a775-603071722901)

        * FLOPS在单精度的时候：
          * 3.0 * 10^9Hz * 8 core * 32 FLOPS/clk = 0.76 TFLOPS
            * 32 FLOPS/clk = 2 FMA * 8个 FP32的 SIMD 运算 * 2乘加融合
              * 2 FMA：
                * 一个时钟周期等于2个浮点运算
              * 4个 FP64 的 SIMD 运算：
                * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/536399ae-621a-406b-92df-0fe294b543b7)




##### （3）FLOPS 在 GPU 中是如何计算的

* 区别：
  * GPU 没有 AVX 这东西
  * 但有大量的 Core 来提高吞吐量
  * 有 Tensor Core 来优化矩阵运算
* 例如：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/01429a08-3e03-4776-a789-cf655798c1c9)

  * 一个SM里面有：
    *  64个处理INT32的CUDA Core
    *  64个处理FP32的CUDA Core
    *  32个处理FP64的CUDA Core
    *  4个处理矩阵计算的的Tensor Core
  * 每一种精度在一个SM中的吞吐量（一个clk可以完成的计算数量）
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/28098927-adc3-4a8b-8b00-166b8dcb3bb8)

* FP64的吞吐量（CUDA Core）：
  * Throughput = 1.41GHz * 108 * 32 * 1 * 2 = 9.7 TFLOPS
    * 频率：1.41 GHz
    * SM数量：108
    * 一个SM中计算FP64的CUDA core的数量: 32
    * 一个CUDA core一个时钟周期可以处理的FP64: 1
    * 乘加: 2
* FP32的吞吐量（CUDA Core）：
  * Throughput = 1.41GHz * 108 * 64 * 1 * 2 = 19.4 TFLOPS
    * 频率：1.41 GHz
    * SM数量：108
    * 一个SM中计算FP64的CUDA core的数量: 64
    * 一个CUDA core一个时钟周期可以处理的FP32: 1
    * 乘加: 2
* FP16的吞吐量（CUDA Core）：
  * Ampere中没有专门针对 FP16 的CUDA core，而是将 FP32 的 CUDA  Core 和 FP64 的 CUDA Core 一起使用来计算 FP16；
  * Throughput = 1.41GHz * 108 * 256* 1 * 2 = 78 TFLOPS
    * 频率：1.41 GHz
    * SM数量：108
    * 一个SM中计算 FP16 的CUDA core的数量: 256
      * **SM中计算FP16的CUDA core的数量是: 256 ( = 32 * 2 + 16 * 4 )**
    * 一个CUDA core一个时钟周期可以处理的FP32: 1
    * 乘加: 2
* INT8的吞吐量（CUDA Core）：
  * Ampere中没有专门针对INT8的CUDA core，而是用INT32的CUDA  Core计算INT8；
  * Throughput = 1.41GHz * 108 * 256* 1 * 2 = 78 **TOPS**
    * 频率：1.41 GHz
    * SM数量：108
    * 一个SM中计算 INT8 的CUDA core的数量: 256
      * **一个SM中计算INT8的CUDA core的数量是: 256 ( = 64 * 4 )**
    * 一个CUDA core一个时钟周期可以处理的FP32: 1
    * 乘加: 2
* INT4的吞吐量：
  * Ampere中没有专门针对INT8的CUDA core，而是用INT32的CUDA  Core计算INT8；
  * Throughput = 1.41GHz * 108 * 256* 1 * 2 = 78 TFLOPS
    * 频率：1.41 GHz
    * SM数量：108
    * 一个SM中计算 INT8 的CUDA core的数量: 256
      * **一个SM中计算INT8的CUDA core的数量是: 256 ( = 64 * 4 )**
    * 一个CUDA core一个时钟周期可以处理的FP32: 1
    * 乘加: 2
* FP16的吞吐量（Tensor Core）：
  * Ampere架构使用的是第三代Tensor Core，可以一个clk完成一个 1024 ( = 256 * 4)个FP16运算。
    * 准确来说是4x8的矩阵与8x8的矩阵的 FMA
      * 256 = 4 * 8 * 8
      * 4 = 一个SM中计算FP16的Tensor core的数量4个
  * Throughput = 1.41GHz * 108 * 4 * 256 * 2 = 312 TFLOPS
    * 频率：1.41 GHz
    * SM数量：108
    * 一个SM中计算 FP16 的Tensor core的数量: 4
    * 一个Tensor core一个时钟周期可以处理的FP16 : 256
    * 乘加: 2



##### （4）CUDA Core vs Tensor Core

* CUDA Core ：
  * 使用一个CUDA Core 计算 C = A * B：
    * 如果使用CUDA Core的话， 需要8次FMA，所以需要8 个clk才可以完成一个c(0,0)
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f970e8b1-b695-4f76-9066-28dab8aa4341)

    * 要完成4 x 8与 8x 4 的计算， 需要 8 * 16 = 128个clk才 可以完成
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/cb18ec0e-d596-46ab-96b1-b0fab45c29bd)

    * 当然，如果我们有16个 CUDA core的话，这些计 算并行，实际上是8个clk。 为了与Tensor Core比较， 这里只用一个CUDA Core
* Tensor Core ：
  * 使用一个Tensor Core 计算 C = A * B：
    * 第一代Tensor Core：
      * Tensor Core不是1个1个的 对FP16进行处理，而是4x4 个FP16一起处理，第一个clk先做A和B的前半段， 结果先存着
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e644b7d6-a45a-4386-afbf-2936f2132fd6)

      * 第二个clk再处理A和B的后半 段，最后和前半段结果做个累 加，完成计算。所以说Tensor  Core处理4x8*8x4的计算只需 要1 + 1 = 2个clk
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d4ac7db0-68b4-482e-9134-15e624a26c18)

    * 第三代Tensor Core：
      * 可以1clk处理 4x8  * 8x8 的操作，也就是说1clk可以处理 256个FP16
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/551a1115-de79-413a-8309-81e549d9eb8a)

  * FP16的吞吐量（Tensor Core）：
    * Ampere架构使用的是第三代Tensor Core，可以一个clk完成一个 1024 ( = 256 * 4)个FP16运算。
      * 准确来说是4x8的矩阵与8x8的矩阵的 FMA
        * 256 = 4 * 8 * 8
        * 4 = 一个SM中计算FP16的Tensor core的数量4个
    * Throughput = 1.41GHz * 108 * 4 * 256 * 2 = 312 TFLOPS
      * 频率：1.41 GHz
      * SM数量：108
      * 一个SM中计算 FP16 的Tensor core的数量: 4
      * 一个Tensor core一个时钟周期可以处理的FP16 : 256
      * 乘加: 2
  * INT8的吞吐量（Tensor Core）：
    * Ampere架构使用的是第三代Tensor Core，可以一个clk完成一个 2048( = 256 * 2 * 4)个INT8运算。
      * 准确来说是4x8的矩阵与8x8的矩阵的 FMA
        * 256 = 4 * 8 * 8
        * 4 * 2 = 一个SM中计算INT8的Tensor core的数量4 * 2个
    * Throughput = 1.41GHz * 108 * 4 * 512 * 2 = 624 TOPS
      * 频率：1.41 GHz
      * SM数量：108
      * 一个SM中计算 INT8 的Tensor core的数量: 4
      * 一个Tensor core一个时钟周期可以处理的INT8: 512 
      * 乘加: 2



#### 4.1.2 Roofline model 与计算密度

Roofline model 在模型部署中的作用仅此于量化，能够帮助我们分析在模型部署时硬件性能卡在哪里了，有没有优化空间。需要理解理解什么叫做Roofline model, memory bound, compute bound， 以及各个layer的计算密度的分类

##### （1）Roofline model 简介

一个衡量计算机软件/硬件性能的一个分析模型。是David Patterson带领的UC  Berkerley的团队与2008年发表的paper中提出的概念。

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4f492163-9224-491a-b5ec-667ac6ed6f88)


Roofline model在模型部署中的意义：

* 针对一个训练好的模型，我们部署的时候可以进行量化、剪枝、蒸馏等优化方法，提升性能
  * 但是在模型已经训练好之后，它的很多框架都是固定的，那么能够优化的地方就很少了，存在局限性
  * 因此在模型创建初期，我们就要尽可能创建计算密度高，同时精度也高的算子，那么再进行量化剪枝的时候就非常不错了
* **可以在 Roofline model 中找到的优化方向**
  * 分析3x3 conv, 5x5 conv, 7x7 conv, 9x9 conv, 11x11 conv的计算效率
    * kernel size越大、计算量就越大、计算资源占用率越大
  * 1x1 conv的计算效率
    * 能够降低模型计算量，让模型轻量化，但是轻量的模型并不一定代表它的计算效率越高、推理时间越短，这和计算密度有关
  * depthwise conv的计算效率
    * 能够降低模型计算量，让模型轻量化，但是轻量的模型并不一定代表它的计算效率越高、推理时间越短，这和计算密度有关
  * 分析目前计算的瓶颈（bottleneck）
    * 分析性能卡在哪里：memory？硬件计算峰值？
  * 分析模型的可以优化的上限
    * 我们不知道模型瓶颈在哪，就不能分析出优化从哪方面切入



##### （2）关键参数

* 计算量（FLOPs）
  * 单位是FLOPs（小写），表示模型中有多少个浮点运算（floating point operations）。 是衡量**模型大小**的标准
* 计算峰值（FLOPS）
  * 单位是FLOPS (也可以是FLOP/s)， 表示计算机每秒可以执行的浮点运算多少（floating point operations）。是衡量**计算机性能**的标准
* 参数量（Byte）
  * 单位是Byte，表示模型中所有的weights(主要在conv和FC中) 的量。是衡量**模型大小**的标准
* 访存量（Byte）
  * 单位是Byte，表示模型中某一个算子，或者某一层layer进行计算时需要与memory产生 read/write 的量。是分析模型中某些计算的**计算效率**的标准之一
  * 计算方法：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/309ece6d-3714-4ca3-a9a7-65920944815d)

    * 所需要的访存量 =  （ kernel size * kernel num + output size * output num） * 4 = 288 Byte = 0.288 KB
      * 4：一般都是用FP32来计算，FP32时32bit，1Byte = 8bit，所以32bit就是4个Byte
  * 陷阱：
    * 参数量和访存量的单位都是 byte，但不一样。conv的**参数量**就是 weight 的大小，**跟input/ouput无关**。 transformer的**参数**会根据**输入 tensor 大小改变**而改变（CNN与Transformer的区别）
* 带宽
  * 单位是Byte/s，全称是 memory bandwidth， 表示的是**单位时间内可以传输的数据量**的多少。是衡量计算机**硬件memory性能**的一 个标准。
    * 影响因素
      * memory clock (GHz)
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2a3b8f22-8c8e-47e3-b16b-42bb39c6c0ce)

      * memory bus width (Byte)
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8ec93ecf-dda5-4991-a39e-21bf4dd7f864)

      * memory channel
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/cfdb1998-0ea5-4530-b467-e80d6fba704b)

  * 计算方法
    * Intel Xeon Gold 6000 (server)
      * => memory bandwidth = 2666 MHz * 8 Bytes * 6 = 128GB/s
        * memory: DDR4-2666
        * memory clock: 2666 MHz
        * memory bus width: 8 Bytes
        * memory channel: 6
    *  NVIDIA Quadro RTX 6000
      * => memory bindwidth = 14 Gbps * 48 Bytes * 1 = 672GB/s
        * memory: GDDR6
        * memory clock: 1750 MHz
        * memory clock effective: 1750 MHz * 8  = 14Gbps
        * memory interface width: 48 Bytes (384 bits)



##### （3）计算密度（Operational intensity）

* 单位是FLOPs/Byte，表示的是传送单位数据可以进行的浮点运算数。
  * 计算密度 = 计算量 / 访存量
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0e9b79c0-ad0a-450e-ae7e-516223f874c8)

* 我们可以通过提高计算密度，让我们的硬件尽量处于饱和状态，从而提高计算效率
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ef6d036a-8dc6-4954-ad61-8e3eb45defb1)




##### （4）硬件性能分析（RTX 3080）

* 以3080为例
  * 硬件资源
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/191a3dc2-7b36-4657-81e9-c6643623530f)

  * 分析计算密度
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ef70418f-7c03-4742-90dc-b82265f4d789)

  * 总结
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7132a3ab-ba10-424d-a010-b3cca2959ea4)

  * 目前我们单独分析了几个layer对计算密度的影响【（5）计算密度的影响因素（FP32的Conv为例）】。但DNN是一个多个layer的组合，所以我们也需要对整个模型进行分析：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/176b9a42-3acd-45fb-8663-579c5acb0456)

  * RTX 3080 Ampere架构中FP32的计算在39.2FLOPs/byte才达到计算饱和：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4e3b5b25-f9dc-40fc-896e-c683f8972b10)

  * 所以这些模型其理论上都没有计算饱和：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6ca4e65e-5d8a-4835-9dba-fa4f33f1feb4)


##### （5）计算密度的影响因素（FP32的Conv为例）

* 1、**kernel size** 的影响
  * 计算公式：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e5241fb2-873a-4199-80b9-f0734fce8226)

  *  **group convolution** 对计算密度的影响：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/166a4f34-53d6-4dae-a613-5081662fd04b)

      * elementwise conv(1x1 conv)的虽然较少了计算量，但是计算密度也很低。随着kernel size增大，计算密度增长率逐渐下降
* 2、**output size** 的影响
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e9424c65-cca9-4455-a2c8-28df038ceadd)

    * 随着output size变大，计算密度的增长率逐渐下降
* 3、**channel size** 的影响
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/53b96629-0801-4082-80b3-e9e508aaa19a)

    * 越大的 channel size 计算密度越高。
* 4、**group convolution** 的影响
  * group：对输入输出分组做卷积的多少组
  * ![image-20240521213017823](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521213017823.png)
    * depthwise虽然降低了计算量，但计算密度也下降的很多
* 5、**tensor reshape** 的影响
  * reshape 矩阵转置的本质并没有计算，只是对数据进行了拷贝和移动
    * 模型中没有tensor reshape
      * ![image-20240521213118429](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521213118429.png)
    * 模型中有3个tensor reshape
      * ![image-20240521213629674](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521213629674.png)
    * 模型中有5个tensor reshape
      * ![image-20240521213642931](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521213642931.png)
  * **tensor reshape** 越多，计算密度越小
* 6、**FC** 的影响
  * 计算公式：
    * ![image-20240521213831477](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521213831477.png)
  * **FC** 对计算密度的影响：
    * ![image-20240521213843917](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521213843917.png)
      * FC的计算密度非常低的原因在于它的大量的访存



##### （6）硬件性能分析（Jetson）

* 以Jetson Xavier AGX Volta为例
  * 硬件资源
    * ![image-20240521214633820](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521214633820.png)
  * 分析计算密度
    * ![image-20240521214718079](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521214718079.png)
  * 总结
    * ![image-20240521214726732](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521214726732.png)
  * 目前我们单独分析了几个layer对计算密度的影响【（5）计算密度的影响因素（FP32的Conv为例）】。但DNN是一个多个layer的组合，所以我们也需要对整个模型进行分析：
    * ![image-20240521214119053](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521214119053.png)
  * Jetson AGX Xavier架构中FP32的计算在10.2FLOPs/byte就计算饱和：
    * ![image-20240521214748433](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521214748433.png)
  * 所以这些模型其实都理论上已经计算饱和：
    * ![image-20240521214324082](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521214324082.png)



##### （7）除了理论上，实际上Roofline影响因素还有很多

* **【重点】**到目前讲的是理论值。然而实际上我们会发现
  * 峰值可能会小于22.4TOPS
  * bandwidth可能会小于137GB/s
  * ![image-20240521215116489](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521215116489.png)
* **需要根据一系列 benchmark 找到部署架构的真实值。**
  * 比如自己写几个计算密集的核函数（减少作为memory cuppy 数据传输用的算子）



### 4.2 模型部署的几大误区

#### 4.2.1 FLOPs 并不能衡量模型性能

* 因为FLOPs只是模型计算大小的单位
* 还需要考虑
  * 访存量
    * 比如一个核函数计算过于复杂，这个核函数都在做reshape这种内存访问的事情，而计算部分占用的并不多
  * 跟计算无关的DNN部分
    * (reshape, shortcut, nchw2nhwc等等) 
  * DNN以外的部分
    * (前处理、后处理这些)
      * 前处理：biliner resize、仿射变换、clop居中等
      * 后处理：YOLO head部的NMS，Tensor Decode等



#### 4.2.2 TensorRT 并不能完全依靠

* TensorRT可以对模型做适当的优化，但是有上限
* 比如
  * <u>计算密度低的**1x1 conv**， **depthwise conv**不会重构</u>
    * 再怎么优化计算密度还是很差，不会好的
  * <u>GPU无法优化的地方会到CPU执行</u>
    * 不是绝对，可以手动修改代码实现部分（比如前处理和后处理），让部分cpu执行转到gpu执行
  * <u>有些冗长的计算，TensorRT可能不能优化，可能为了优化添加一些多余的操作</u>
    * 比如类似于量化的时候添加**reformatter**这种算子（TensorRT为了达到Tensor量化形状匹配时添加）
    * 直接修改代码实现部分
  * <u>存在TensorRT尚未支持的算子（或者效率不高）</u>
    * 可以自己写plugin，可以用 cuBLAS、catlas写一个高效的plugin
  * <u>TensorRT不一定会分配Tensor Core</u>
    * trtexec 推理引擎创建的时候
    * 因为TensorRT kernel auto tuning会选择最合适的kernel



#### 4.2.3 CUDA Core 与 Tensor Core 的区别

* 有的时候TensorRT并不会分配Tensor Core
  * kernel auto tuning自动选择最优解
  * 📌【面试】所以有时会出现类似于**INT8的速度比FP16反而慢**了
    * FP16量化的时候，TensorRT会找一些能够在Tensor Core上跑的一些Kernel核函数
    * 但当我们给它设定成INT8的时候，比如有一些算子不支持INT8，或者有些计算在转为INT8的时候会添加多余的操作（例如在QAT的时候会QDQ，QDQ做融合会添加很多其他操作，让模型变得复杂，TensorRT觉得添加那么多操作后再Tensor Core上执行效率并不是那么高，那就自动fullback到CUDA Core上去执行）
    * 这时就会出现FP16在CUDA Core上，而INT8就跑到CUDA Core上的现象
  * **使用Tensor Core需要让tensor size为8或者16的倍数**（记住就好）
    * 8的倍数：fp16精度
    * 16的倍数：int8精度



#### 4.2.4 不能忽视 前处理/后处理 的overhead

* 对于一些轻量的模型，相比于DNN推理部分（几毫秒），前处理/后处理可能会更耗时间
  * 因为有些前处理/后处理的复杂逻辑不适合GPU并行
* 然而有很多种解决办法
  * 可以把前处理/后处理中可并行的地方拿出来让GPU并行做，不用openCV做
    * 比如RGB2BGR, Normalization, resize, crop, NCHW2NHWC
  * 可以在CPU上使用一些针对图像处理的优化库
    * GPU在忙的时候，比如视频推理的时候，针对每帧的overlaping，前一帧DNN部分推理计算完之后直接让后一帧开始推理计算，而让CPU去做第一帧的后处理。让前处理-DNN-后处理实现重叠操作
    * 比如Halide，使用Halide进行blur, resize, crop, DBSCAN, sobel这些会比 CPU快
* 并不是能GPU加速的地方就GPU加速
  * 需要考虑GPU占用率



#### 4.2.5 对使用TensorRT得到的推理引擎做benchmark和profiling

* 使用TensorRT得到推理引擎并实现infer只是优化的第一步
* 需要使用NVIDIA提供的benchmark tools进行profiling
  * 分析模型瓶颈在哪里
  * 分析模型可进一步优化的地方在哪里
    * 比如提高访存效率（带宽）或者计算量？
  * 分析模型中多余的memory access在哪里
    * 比如reformate
  * 可以使用：
    * nsys, nvprof, dlprof, Nsight这些工具



### 4.3 模型部署优化-量化

#### 4.3.1 理解量化诞生的背景与意义

#### 4.3.2 量化的基本算法与对称/非对称量化

#### 4.3.3 量化粒度与精度/效率的关系

#### 4.3.4 量化校准算法比较

#### 4.3.5 PTQ 量化以及 layer-wise 敏感度分析

#### 4.3.6 QAT 量化以及 Q/DQ 节点与算子的融合

#### 4.3.7 常见的量化技巧与正确的量化思路



### 4.4 模型部署优化-剪枝

#### 4.4.1 Channel purning 算法与 L1-Norm 的关系

#### 4.4.2 Fine-grained structured sparse pruning

#### 4.4.3 分析 Sparse Tensor Core 硬件层面处理剪枝
