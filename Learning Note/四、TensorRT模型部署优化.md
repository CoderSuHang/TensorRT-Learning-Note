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
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/32436ac1-a5bd-4ac7-9cf9-47f07edba8d5)

    * depthwise虽然降低了计算量，但计算密度也下降的很多
* 5、**tensor reshape** 的影响
  * reshape 矩阵转置的本质并没有计算，只是对数据进行了拷贝和移动
    * 模型中没有tensor reshape
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e0b0dd38-7f83-4db3-8f06-06582a2b5577)

    * 模型中有3个tensor reshape
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/59d1ba73-1b64-4a82-b03f-ca9b39458836)

    * 模型中有5个tensor reshape
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5b28fca6-8a86-4733-816f-db13efe31f2f)

  * **tensor reshape** 越多，计算密度越小
* 6、**FC** 的影响
  * 计算公式：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f488fb26-283e-413a-82a0-0308e36669f9)

  * **FC** 对计算密度的影响：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1687c4fa-6765-4596-84e2-83cb8c10e9e1)

      * FC的计算密度非常低的原因在于它的大量的访存



##### （6）硬件性能分析（Jetson）

* 以Jetson Xavier AGX Volta为例
  * 硬件资源
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4c4cf39a-45dc-455e-a798-b46a21957d20)

  * 分析计算密度
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f4dfaeb4-472c-4fd9-91a1-6360e636c975)

  * 总结
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4c693d83-5621-48be-bcd7-135f8c7894af)

  * 目前我们单独分析了几个layer对计算密度的影响【（5）计算密度的影响因素（FP32的Conv为例）】。但DNN是一个多个layer的组合，所以我们也需要对整个模型进行分析：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/dcdb6fc5-ee45-460d-9f2a-f02a4fa5d2f1)

  * Jetson AGX Xavier架构中FP32的计算在10.2FLOPs/byte就计算饱和：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7486da1c-5ce6-4153-a635-87b43699c71d)

  * 所以这些模型其实都理论上已经计算饱和：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8ed1eba2-3186-46a1-a78a-6a0775eca642)




##### （7）除了理论上，实际上Roofline影响因素还有很多

* **【重点】**到目前讲的是理论值。然而实际上我们会发现
  * 峰值可能会小于22.4TOPS
  * bandwidth可能会小于137GB/s
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5f0664ae-a047-49ea-92a5-d52137e0ad58)

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

背景

* DNN模型的大小，几乎在以每年**10**倍的FLOPs在增长
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8e64966d-d94d-474e-9308-6d9a61f1a3d1)


* 相反，硬件的性能却以仅每年0.74倍FLOP/s的速度增长
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b9d029cd-3ca8-4a3d-8b6b-95f5918e9c97)

* 相比于模型的发展，硬件的发展速度很慢。即便硬件有了，还需要有相对应的的编译器。有了基本的编译器后，还需要有编译器的优化（TensorRT 3.x~8.x的演变），还需要有一套其他的SDK。

意义：

* 所以，大家一般会考虑如果用现有的硬件基础上**减少模型计算量**、**增大模型计算密度**等等。所以针对 这些需求，就有了**“量化(quantization)”**，“剪枝(Prunning)”等这些优化方法。



#### 4.3.2 量化的基本算法与对称/非对称量化

##### （1）量化简介

1、模型量化是通过减少模型中计算精度从而**减少模型计算所需要的访存量**，进而进一步提高计算密度的一种方法。计算精度可以分为FP32, FP16, FP8, INT8,  INT32, TF32这些

2、量化针对的是：

* activation value（模型的激活值，例如输入输出这也Tensor value）
* weight（权重）

3、所以一般来说我们会对**conv**或者**linear**这些计算密集型算子进行量化

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e4fe8faa-6239-4d17-a7cb-a1629a2b3f19)

  * 量化和反量化的过程

##### （2）量化会出现什么问题

数据的动态范围：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/24a1a671-8073-4af7-bf14-4974e787ada5)


仅仅用256种数据去表现FP32的所有可能出现的数据，有可能会造成**表现力下降**。

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2d2b8286-73a3-4ae2-85dd-5dbc80aa174e)


如果能够比较完美的用这256个数据去最大限度的表现FP32的 原始数据分布，是量化的一个很大挑战。换句话说，就是如何合理的设计这个**dynamic  range**是量化的**重点**



##### （3）量化的基本原理：映射和偏移

倘若想把R中的数据用Q来表示，如何做？

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e3c2487a-eb88-45da-87a3-f2c9796a9916)


【方法一】

* 1、根据R和Q中x和y可以取的最大值和最小值，计算得到一个**缩放比**(ratio)：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6a1e412b-cb39-47d8-a818-ae9bc3fe78ee)

* 2、以及缩放后的R要在Q的范围中显示，所需要的偏移量(distance):
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/900dd738-1775-4095-9b73-97fd7a72f1c3)

* 3、最终，通过ratio和distance的到x和y的关系
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/30f2f3d9-2b8d-485f-9fad-b8e404bdfdf5)

* 4、演示：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/59edf895-daeb-4fa5-b608-14f9d8a236a7)

  * 通过ratio和distance我们可以这么理解：
    * Q中每一个元素可以代表R中每5个元素，并且偏移量是20
* 5、问题：
  * 如果说可以通过上面的公式将R中的数据映射到Q中的话，那么我们按照下面的公式反着计算的话，是不是就可以通过Q中的数据得到R呢？
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/07b8b9e2-85fa-4385-9ddd-1c8c3e2a94fc)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/acbb43fe-00fa-4d04-874a-7ad664981614)

      * 相比于原本的101个R中的数据，如今我们只能够得到R中21个数据，比如说-96， -93， -81是无法得到的
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/387db2cc-d44d-4ea4-8573-e12d37432991)

        * 很明显，虽然下面的4个example中数据都呈现-100~0中，但是由于数据的分布形式不同，如果我们统一都用一种ratio和distance的话，会有很大的误差
          * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/79a1c21b-e2b0-4d26-b921-37dad20ca472)

        * 所以，为了能够让R到Q的映射合理，以及将Q中的数据还原为R时误差能够控制到最小，我们需要**根据R中的数据分布**合理的**设计这个ratio和distance**
          * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b971a1ba-0076-4973-8e34-78e596c6990c)




##### （4）基本术语

* R是一组FP32的数据，能够表现的数据种类有很多，大约是 2^(32) 种(4亿):
  * 范围是: −1.2 ∗ 10^(−38) ~ 3.4 ∗ 10^(38)
* Q是一组INT8的数据，只能够表现2^(8)种数据(256)
  * 范围是：-128 ~128 or 0 ~ 255
* R到Q的映射的**缩放因子scale**的计算公式为
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6cd0f02e-c155-4a0b-9d58-00fa95ffaaa0)

* R缩放之后映射到Q时，所需要的**偏移量z**为
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bfddf65d-78f8-413e-b34e-f13e4a9e9bd6)

* 这样R中每一个元素转移到Q的过程称为**量化**(Quantization)，公式是
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/04d892cd-9313-410e-9d34-dee708345fc0)

* 将Q空间中一个元素转换回R的空间的过程为**反量化**(Dequantization)，公式是
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/dd89dc3d-8cd5-44ad-8d21-ba653f0f7b59)




##### （5）对称映射，非对称映射

* 根据R和Q的 dynamic range 的选择以及 mapping 的方式，我们可以分为，对称映射(symmetric  quantization)，以及非对称映射 (asymmetric quantization)
  * 对称映射(symmetric  quantization)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/03198ddd-18d9-4a93-8858-454d529e5c26)

    * 对称量化中量化前后的0是对齐的， 所以不会有偏移量(z, shift)的存在， 这个可以让量化过程的计算简单。 NVIDIA默认的mapping就是对称量化，因为快
  * 非对称映射 (asymmetric quantization)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/edcc5f89-757c-45f4-aa41-bf17d0bdc137)




##### （6）量化粒度

* 量化中非常重要的概念: Quantization Granularity(量化粒度)
  * 指的是对于一个Tensor，以多大的粒度去共享scale和z，或者dynamic range，具体选哪一个粒度好会很大程度影响性能和精度！包括：
    * per-tensor quantization（一个tensor中所有的 element共享同一个 dynamic range）
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/45ac3e74-6f06-4750-b2cd-54cbc8dfc247)

    * per-channel quantization（一个tensor中每一个layer都有一个自己的dynamic  range）
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a3273e0b-c033-4244-852d-254c63210754)

    * per-element quantization（一个tensor中每一个element都有一个自己的dynamic range。 也可以叫做element-wise  quantization）
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0c3d281e-5c80-4563-a35d-7e8b21f2bc92)




##### （7）校准

* 量化中另外一个非常重要的概念：Calibration(校准)
  * 对于一个训练好的模型，**权重是固定**的，所以可以通过一次计算就可以得到每一层的量化参数。
  * 但是activation value(激活值)是**根据输入的改变而改变**的。所以需要通过类似于统计的方式去寻找对于不同类型的输入的不同的dynamic range。这个过程叫做校准。
  * 跟量化粒度一样，不同的校准算法的选择会很大程度影响精度！
* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8a6861fb-0194-4fe3-8472-369348d257d5)




##### （8）PTQ, QAT

* 根据量化的时机，一般我们会把量化分为
  * PTQ(Post-Training Quantization)，训练后量化
  * QAT(Quantization-Aware Training)，训练时量化
* PTQ一般是指对于训练好的模型，通过 calibration 算法等来获取 dynamic range 来进行量化。
* 但PTQ不会更新权重weights，量化普遍上会产生精度下降。所以QAT为了弥补精度下降，在学习过程中通过Fine-tuning权重来适应这种误差，实现精度下降的最小化。
* 所以一般来讲，QAT的精度会高于PTQ。但并不绝对。详细在下下下一小节讲。
* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/67909d1d-4958-4ffb-8d19-c12d77806fd4)




##### （9）有关量化学习的激活函数

* 量化学习是一个Fine-tuning的过程。那么选取什么样子的激活函数会更好呢？
  * 我们可以结合量化的特性去思考。我们希望整个学习过程让权重或者激活值控制在某个区域范围内，所以我们需要实现某种Clipping。推荐两个激活函数：
    * PACT(Paramertized Clipping Activation Function)
      * 对于PACT的介绍，推荐阅读一下IBM的论文
    * ReLU6
      * PACT(Paramertized Clipping Activation Function)



#### 4.3.3 量化粒度与精度/效率的关系

##### （1）量化粒度

量化中非常重要的概念: Quantization Granularity(量化粒度)

* 指的是对于一个Tensor，以多大的粒度去共享scale和z，或者dynamic range，具体选哪一个粒度好会很大程度影响性能和精度！包括：
  * per-tensor quantization（计算容易）
    * 一个tensor中所有的 element 共享同一个 dynamic range
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0eaa5ee9-b81c-4c59-af89-e267ecc49d58)

  * per-channel quantization（计算中等）
    * 一个tensor中每一个layer都有一个自己的dynamic  range，当我们遇到每一个channel动态范围差别太大的时候就会用到。
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7f4bede4-0d0f-4521-900a-d0c0faf6b0c6)

  * per-element quantization（计算麻烦）
    * 一个tensor中每一个 element 都有一个自己的 dynamic range。 也可以叫做 element-wise  quantization
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c1b50620-7423-45e2-8bb0-1d52b22123fb)




##### （2）Per-tensor & Per-channel量化

* Per-tensor量化
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/88c09d7a-b87e-4015-8e31-f5612c8fe85f)

  *  (优点）低延迟: 一个tensor共享同一个量化参数
  *  (缺点）高错误率: 一个scale很难覆盖所有FP32的 dynamic range
*  Per-channel (layer)量化
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/840cc776-3d9b-4468-b0b5-d2bae675fe47)

  *  (优点）低错误率: 每一个channel都有自己的scale
  *  (缺点）高延迟: 需要使用vector来存储每一个channel的scale



##### （3）量化粒度选择的推荐方法

* **（重点）**从很多实验结果与测试中，对于 weight 和 activation values 的量化方法，一般会选取
  * 对于activation values，选取per-tensor量化
  * 对于weights，选取per-channel量化
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3bf54c77-ac47-4894-83f7-85e6e57caa68)

* 为什么**weight**需要per-channel呢？主要是因为
  * BN计算与线性计算的融合（BN folding）
    * 线性变化 𝑦 = 𝑤 ∗ 𝑥 的BN folding可以把BN的参数融合在线性计算中。但是BN的可参数是per-channel的。如果weights用per-tensor的话，会掉精度。
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c849382f-3207-402c-8568-386813cbb6f6)

  * depthwise convolution
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/627edd95-77f0-4fd1-a6b8-7ae8906e567c)

    *  depthwise convolution 中 kernel 的 channel size 是1，每一个 kernel 针对输入的对应的 channel 做卷积。
    * 所以每一个 channel 中的参数可能差别会比较大。如果用 per-tensor 的话容易掉精度比较严重
    * 例如下面量化精度效果：
      * MobileNet: 
        * MobileNet: (FP32) 71.88
        * MobileNet: (int8 Per-channel weight quantization)  71.56
        * MobileNet: (int8 Per-tensor weight quantization)  66.88
      * EfficientNet: 
        * EfficientNet: (FP32) 76.85
        * EfficientNet: (int8 Per-channel weight quantization)  76.72
        * EfficientNet: (int8 Per-tensor weight quantization)  12.93

* **（重点）**目前的TensorRT已经默认对于Activation values选用Per-tensor，Weights选用 Per-channel，这是他们做了多次实验所得出的结果。很多其他平台的SDK可能不会提供一些默认的量化策略，这是我们需要谨慎选择，尽快找到掉点的原因。



#### 4.3.4 量化校准算法比较

##### （1）校准简介

* 量化中另外一个非常重要的概念：Calibration(校准)
  * 对于一个训练好的模型，**weights（权重）是固定**的，所以可以通过一次计算就可以得到每一层的量化参数。
  * 但是**activation value（激活值）**是**根据输入的改变而改变**的。所以需要通过类似于**统计的方式**去寻找**对于不同类型的输入的不同的dynamic range**。这个过程叫做**校准**。
  * 跟量化粒度一样，不同的校准算法的选择会很大程度影响精度！
* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e0ddb8ce-18d8-45d9-ac5f-15152cd012fd)

  * 横向tensor FP32的值，纵向每个数出现的次数
  * 做量化的时候，我们一般可以用max、entropy、percen等方法取FP32的动态范围，



##### （2）Calibration dataset

* 校准一般会在PTQ训练后量化的时候出现，需要用到校准数据集
  * 针对不同的输入，各层 layer 的 input activation value 都会有不同的分布和取值。大数据集的差别比较大。
  * 我们需要通过训练数据集中的一部分数据来尝试表征整个数据集的分布。
  * 这个小数据集就是calibration dataset。一般往往很小，但需要尽量有整体的表征



##### （3）Calibration algorithm

*  calibration的过程一般是在模型训练以后进行的，所以一般与PTQ(*)搭配使用。整体的流程就是:
  * 在 calibration dataset 中做一次FP32的推理
  * 以 histogram 的形式去统计每一层的floating point的分布
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bccc9315-0688-4642-b2eb-4730569c8a2c)

    * （注意，因为activation value是per-tensor quantization）
  * 寻找能够表征当前层的 floating point 分布的 scale
    * 这里会有几种不同的算法，比较常见的有
      * Minmax calibration
      * Entropy calibration
      * Percentile calibration
    * (以上这些过程TensorRT都已经帮我们封装好了，可以拿来直接用)
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/21070d19-6483-46c9-a61f-1e23ed7d1364)


* Minmax calibration
  * 把FP32中的最大值和最小值全部考虑进去
  *  FP32->INT8的scale需要能够把 FP32 中的最大最小值都给覆盖住。
    * 如果 floating point 的分布比较离散， 各个区间下的分布都比较均匀，minmax是个不错的选择
    * 然而，如果只是极个别数据分布在这种地方的话，会让dynamic range变得比较 稀疏，不适合用minmax
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/28b1cfa8-b15e-4515-86b8-38f09799aa87)

* Entropy calibration
  * 通过计算 KL 散度，寻找一种 threashold，能够最小化量化前的 FP32 的浮点数分布于 INT8 的量化后整形分布
    * 目前 TensorRT 使用默认的是 Entropy  calibration。一般来讲使用entropy  calibration精度可以比较好
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/df360ff1-88f0-4d74-9fb3-ceb7bbada1d5)

* Percentile calibration
  * 如同字面意思，表示的是FP32中**占据 99.99% 的浮点数**参与量化。
    * 这样可以避免极个别特殊点（误差）参与量化，导出量化出现问题
    * Percentile有99.9%, 99.99%,  99.999%等等
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2b7c997e-e4de-4fcd-a38a-dd43f90d23cd)

* 如何选择calibration algorithm
  * **weight** 的calibration，选用 minmax
    * weight权重信息少，并且重要，所以可以全部截取
  * **activation** 的calibration，选用 entropy 或者 percentile
    * 激活值差异很大
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3962be26-569c-47c0-8665-63fe589af628)




##### （4）calibration dataset与batch size的关系

在使用calibration dateset中构建histogram是需要注意的一个点：calibration时的batch size（一个batch中有几张图片）会影响精度。 更准确来说会影响histogram的分布，这个跟TensorRT在构建浮点数的histogram的算法有关：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/28607601-b6ec-421a-8e1e-9f268cc2ad10)

* 上面的说法表明：在创建 histogram 直方图的时候，如果出现了大于当前 histogram 可以表示的最大值的时候，TensorRT会直接平方当前histogram的最大值，来扩大存储空间
  * 如果batchsize=1，最后一个batch的浮点数很大，那么最终的histogram会呈现什么形状？
    * 这里以batchsize=8为例
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/eff3d388-cd71-47f5-b27c-14065f1a2215)

      * 这时 histogram 的后半段很稀疏，甚至没有数据。
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c5fd458d-5583-4bd9-aa91-f3793a6d7d60)

      * 在量化的时候会根据这个直方图来将 FP32转为INT8，很显然这块领域是多余的
  * 如果batchsize=16，但每一个batch size的数据分布很均匀，histogram会呈现什么形状？
    * 我们希望每一个batch里面的数据比较均匀， 让比较大的数据出现的时候，histogram的范围已经能够表现它了。
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/732a0667-4817-415e-97a7-01dba91325c6)

    * 当2.4出现的时候，如果之前已经出现过1.54，那么hisogram的range不需要改变。否则range的最大值会变成5.76
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/49e377bc-60c8-41b0-aa35-c8607461b5c0)

      * 总的来讲，calibratio的batch size越大越 好，但不是绝对的
  * 如果模型的鲁棒性很强，batchsize=1和 batchsize=16/32/64/128 的区别会有吗
    * 有的，不管鲁棒性强不强，都尽量以大的 batch size 为主
  * 如果模型的鲁棒性很强，calibration dataset = 1000/500/100 的区别会有吗
    * 关系不大，建议1000起步



#### 4.3.5 PTQ 量化以及 layer-wise 敏感度分析

##### （1）PTQ, QAT简介

* 根据量化的时机，一般我们会把量化分为
  * PTQ(Post-Training Quantization)，训练后量化
    * PTQ一般是指对于训练好的模型，通过 calibration 算法等来获取 dynamic range 来进行量化。但PTQ不会更新权重weights，量化普遍上会产生精度下降。
  * QAT(Quantization-Aware Training)，训练时量化
    * 所以QAT为了弥补精度下降，在学习过程中通过Fine-tuning权重来适应这种误差，实现精度下降的最小化。
    * 所以一般来讲，QAT的精度会高于PTQ。但并不绝对。
* PTQ流程
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/118e0252-d7ac-49ad-9b7b-dc424eaa4071)

    * 1、准备一个校准集，大概是整个数据集的10%左右
    * 2、把数据集放在训练好的模型上
    * 3、统计每一层的信息
    * 4、对每一层进行计算，获得每一层量化的scale
    * 5、最后拿scale进行量化模型
* QAT流程
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d16cd20b-fb5d-48fa-8786-48b852dccb22)

    * 1、准备一个训练好的模型
    * 2、对模型添加QDQ的节点（量化和反量化的节点）
    * 3、结合QDQ节点来通过Fine-tuning更新权重
    * 4、整个过程存储scale这些信息
    * 5、最后拿scale进行量化模型



##### （2）PTQ是什么

* PTQ(Post-training quantization)也被称作隐式量化(implicit quantization)。
  * 我们并不显式的对算子添加量化节点(Q/DQ)，calibration之后TensorRT根据情况进行量化
  * 例如：
    * trtexec在选择参数进行fp16或者int8指定的时候，使用的就是PTQ。(int8的时候需要指定calibration dataset)。很方便使用，但是我们需要先理解PTQ的利弊



##### （3）PTQ优缺点分析

* 优点：
  * 方便使用，不需要训练。可以在部署设备上直接跑
* 缺点
  * 1、精度下降
    * 量化过程会导致精度下降。但PTQ没有类似于QAT这种fine-tuning的过程。所以**权重不会更新来吸收这种误差**
  * 2、量化不可控
    * TensorRT会权衡量化后所产生的新添的计算或者访存， 是否用INT8还是FP16。
    * TensorRT中的kernel autotuning会选择核函数来做FP16/INT8的计算。来查看是否在CUDA core上跑还是在Tensor core上跑
    * 有可能FP16是在Tensor core上，**但转为INT8之后就在CUDA core上了**
  * 3、层融合问题
    * 量化后有可能出现之前可以融合的层，不能融合了（因为量化只有有些层不支持FP16或INT8）
    * 量化会添加reformatter这种更改tensor的格式的算子，如果本来融合的两个算子间添加了这个就不能被融合了
    * 比如有些算子支持int8，但某些不支持。之前可以融合的，但因为精度不同不能融合了
  * 如果INT8量化后速度反而会比FP16/FP32要慢，我们可以从以上的2和3去分析并排查原因


##### （4）化中的sensitive analysis

* 从精度分析的角度去弥补PTQ的精度下降，我们可以进行layer-wise的量化分析。这种方法被称作**layer-wise sensitive analysis**
  * 例如 EfficientNetb0的模型框架
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1dd5d122-822d-4c38-ba7b-d51f7d5e8c6f)

    * 对EfficientNetb0的各层进行量化分析，寻找影响精度的层
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ce52ac0c-d18c-462a-abfd-d48a2f21505a)

* 需要注意的点【输入输出层附近尽量不要INT8量化，因为这里的数据又少又重要】
  * 普遍来讲，模型框架中会有一些层的量化对精度的影响比较大。我们管它们叫做敏感层(sensitive layer)。
    * 对于这些敏感层的量化我们需要非常小心。尽量用FP16。敏感层一般靠近模型的输入输出。
      * 靠近输入属于敏感层：
        * channel还比较小，每一个位置所具有的特征量可能还比较分散。建议FP16
      * 模型中间部分：
        * 计算比较密集，特征量也比较大。建议INT8
      * 靠近输出属于敏感层：
        * 后处理的部分对这里的tensor的数据要求比较高。建议FP16
      * 最终模型的推理引擎是FP16+INT8精度的
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/aff86213-034c-4230-b158-006e6881ea2b)




##### （5）学习使用Polygraphy

争对敏感度分析（sensitive analysis），NVIDIA提供了 polygraphy 分析工具，能够分析并查找模型精度下降并且影响比较大的地方，**做TensorRT量化必须要掌握的工具**，能够实现的功能如下：

* onnxruntime与TensorRT engine的layer-wise的精度分析
* 输出每一层layer的权重histogram
* 截取影响整个网络中对精度影响最大的子网，并使用onnx-surgeon单独拿出来



##### （6）FP16/INT8对计算资源的利用

📌我们在做量化后，我们无法指定将量化后的conv或者gemm放在Tensor core还是在CUDA core上计算，这些是TensorRT在帮我们选择核函数的时候自动完成的。

查看一般有三个方法：

* 【1】使用dlprof
  *  DLProf (Deep learning Profiler)工具可以把模型在GPU上的执行情况以TensorBoard的形式打印出来，分析TensorCore的使用情况。感兴趣的可以查看一下。但需要注意的是，**DLProf不支持Jetson系列的Profile**。对于Jetson，我们可以使用Nsight system或者trtexec
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/421c0594-7d67-463e-b862-5e0174da9cc3)

* 【2】使用nsight system
  * 如果是利用Nsight system的话，我们可以查看到哪一个kernel的时间占用率最高，之后从kernel的名字去推测这个kernel是否在用Tensor Core。(从kernel名字推测kernel的计算设备需要经验)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/dbf96202-c948-441e-9bec-d35838c291cd)

    * 从kernel名字推测可以从kernel中的关键字去猜，比如：
      * h884 = HMMA = FP16 TensorCore
      * i8816 = IMMA = INT8 TensorCore
      * hcudnn = FP16 normal CUDA kernel (without TensorCore)
      * icudnn = INT8 normal CUDA kernel (without TensorCore)
      * scudnn = FP32 normal CUDA kernel (without TensorCore)
* 【3】使用trtexec



#### 4.3.6 QAT 量化以及 Q/DQ 节点与算子的融合

##### （1）QAT简介

QAT(Quantization Aware Training)也被称作显式量化。

* 我们明确的在模型中添加Q/DQ节点 (量化/反量化)，来控制某一个算子的精度。
* 并且通过fine-tuning来更新模型权重，让权重学习并适应量化带来的精度误差
* QAT的核心就是通过添加fake quantization，也就是Q/DQ节点，来模拟量化过程。



##### （2）Q/DQ简介

**1、Q/DQ node也被称作fake quantization node**

* Q是用来模拟fp32->int8的量化的scale和shift(zero-point)
* DQ是int8->fp32的反量化的scale和shift(zero-point)
* QAT通过Q和DQ node里面存储的信息对fp32或者int8进行线性变换

**2、Q/DQ节点的插入示意：**

* 没有QAT的默认onnx模型架构
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a8c8a6fe-3c7b-4996-bf83-e4b0463ae88a)

* 为带有QAT的onnx模型架构
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f00745c7-699c-4f7b-9bf4-719d06efae4e)

  * 添加Q/DQ节点模拟量化之后，如果出现误差，会让Conv更行权重weight来适应

**3、Q/DQ公式：**

* 参数说明
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9807af5e-95f6-467f-b0bb-92e70c01b71c)


* 那么Q的公式可以理解为
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/25e5d892-1aaa-414f-90b2-358fbcbba1b3)

    * clip是截取功能
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f31dc1e8-d881-48b2-a3f7-63ecee66f234)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/150318bd-e17c-43cb-b446-195cd3ded64d)

* DQ的公式可以理解为
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9334c375-bccb-45af-b1ac-4f3cf9a0e2fb)


##### （3）可量化层的计算

**1、DQ + fp32精度的op融合**

对于一个线性计算的op(conv或者linear)

* fp32精度的op的计算简化成
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5cd1adb4-57e5-4aad-a9d5-ffcc3ffc8338)

* 既然x和w是fp32的，那么我们也可以这么表示
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/97d40e3f-80ea-4a92-a77a-fb356f25b013)

    * 这里以NVIDIA采用的对称量化量化与反量化计算为例，计算过程没有涉及zero-shift
  * 展开
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/222ea089-a299-44ff-b754-5196949bc1fe)

* 因为计算量的主要是𝑤𝑞 ∗ 𝑥𝑞,是int8计算，所以我们可以把这个公式写成：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/025a01b8-cc73-4112-9c5e-d1e9303aaad9)

  * 所以我们知道DQ + fp32精度的op可以拼成一个int8精度的op，但输出都是FP32
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/70f63bfe-808e-4ea1-a1be-252e1eb084cb)


**2、DQ + fp32精度op + Q的融合**

* 下一个Q的计算
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2a7efabf-1fc2-43dc-95d5-d8d4484187eb)

    * 这里的𝑥′是来自于上一 层的输出，是fp32
* 由于𝑥′是来自于上一层计算，可以把𝑥′展开
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/495ae29b-d97a-4df1-9c3e-100acf2f3360)

* 我们可以看到这个依然是一个线性变化。所以说DQ + fp32精度OP + Q可以融合在一起凑成一个int8的op，所以我们可以把这个公式替换成：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/03aab722-f97d-4309-b539-a4077f3436b8)

* 我们称这个op或者layer为quantizable layer，翻译为可量化层
  * 这个可量化层的**输入**和**输出**都是int8
  * 计算的主体也是int8，可以**节省带宽**的同时，**提高计算**效率
* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/11e71f0c-d6b4-49b8-892e-70aa70947e13)




**3、融合图解**

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9952a564-910d-4bfc-9acf-625610b2f331)


我们知道conv和Relu是可以融合在一起成为ConvReLU算子，同时根据之前的公式和图，我们知道：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bdbbde39-c8a0-4e64-9304-8824e28de84f)

  * DQ和fp32精度的conv组合在一起，可以融合成一个int8精度的conv
  * fp32精度输出的conv和后面的Q也可以融合在一起，输出一个int8精度的activation value
* 将这些虚线包围起来的算子融合在一起，用一个int8的op来替换后，整个网络就会变成这个样子:
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5a53d7a9-f5ba-48eb-9dff-60442824d2b2)

  * 新生成的QConvRelu以及Qconv是int8精度的计算，速度很快并且TensorRT会很大几率分配tensor core执行这个计算。这个就是TensorRT中对量化节点的优化方法之一。



##### （4）QAT的工作流

理解了Q/DQ再去看QAT就非常容易了。QAT是一种Fine-tuning方式，通常对一个pre-trainedmodel进行添加Q/DQ节点模拟量化，并通过训练来更新权重去吸收量化过程所带来的误差。添加了Q/DQ节点后的算子会以int8精度执行

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7ecdb754-e64a-4f1a-8aab-aeac342d6e2a)

* pytorch支持对已经训练好的模型自动添加Q/DQ节点。
  * 详细可以参考：https://github.com/NVIDlA/TensorRT/tree/main/tools/pytorch-quantization



#### 4.3.7 QAT常见的量化技巧与正确的量化思路

##### （1）TensorRT中QAT的层融合的技巧

TensorRT对包含Q/DQ节点的onnx模型使用很多图优化，从而提高计算效率。主要分为：

* Q/DQ fusion
  * 通过层融合，将Q/DQ中的线性计算与conv或者linear这种线性计算融合在一起，**实现int8计算**
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e54c5be2-53c8-4a35-a255-2e88b1fff293)

* Q/DQ Propagation
  * 将Q节点尽量往前挪，将DQ节点尽量往后挪，**让网络中int8计算的部分变得更长**
  * Max Pooling与Q/DQ的propagation：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b74c7527-2863-4a55-a2eb-77782a73f20c)

    * 由于maxpooling的结果在量化前后是没有变化，所以我们可以把fp32的maxpool节点转为int8的maxpool，从而达到加速
  * ❗【注意】有的时候我们发现TensorRT并没有帮我们做到最好，这个时候我们可以使用TensorRT API来手动修改



##### （2）QAT的学习误差

* 主要是训练weight来学习误差
  * Q/DQ中的scale和zero-point也是可以训练的。通过训练来学习最好的scale来表示dynamic（训练好的weights分布能够更好的用scale表现出来）
* 没有PTQ中那样人为的指定calibration过程、
  * 不是因为没有calibration这个过程来做histogram的统计
  * 而是因为QAT会利用fine-tuning的数据集在训练的过程中同时进行calibration
  * 这个过程是我们看不见的。这就是为什么我们**在pytorch创建QAT模型的时候需要选定calibration algorithm**
    * 如何选最好的calibration：
      * 使用不同的calibration algorithm进行QAT的精度比较。粗体表示使用PTQ中可以达到最好的calibration algorithm
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ba971beb-0130-4de0-aff3-d9db202bdb7d)

    * 对于activation value的scale进行学习的过程(上为forward，下为backward）
      *  ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/61a044f5-21f9-43b5-b954-bf6f542ee837)




##### （3）我们在部署过程中应该按照什么样子的流程进行QAT

* **没有必要盲目的使用QAT，在使用QAT之前先看看PTQ是否已经达到了最佳。**可以按下图进行量化测试：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/93d99419-e4c6-41af-8167-12798ca2ba30)


1. 先进行PTQ
   1. 从多种calibration策略中选取最佳的算法
   2. 查看是否精度满足，如果不行再下一步。
      1. 普遍来讲，量化后精度下降控制在**相对精度损失<=2%**是最好的。
2. 进行partial-quantization
   1. 通过layer-wise的sensitve analysis分析每一层的精度损失
   2. 尝试fp16 + int8的组合
   3. fp16用在敏感层(网络入口和出口)，int8用在计算密集处(网络的中间)
   4. 查看是否精度满足，如果不行再下一步。
   5. **注意，这里同时也需要查看计算效率是否得到满足**
3. 进行QAT来通过学习权重来适应误差
   1. 选取PTQ实验中得到的最佳的calibration算法
   2. 通过fine-tuning来训练权重(大概是原本训练的10%个epoch)
   3. 查看是否精度满足，如果不行查看模型设计是否有问题
   4. (注意，这里同时也需要查看层融合（Q/DQ fusion、Propagation）是否被适用，以及Tensor core是否被用)


### 4.4 模型部署优化-剪枝

#### 4.4.1 模型剪枝的概念

##### （1）模型剪枝简介

模型剪枝是不同于量化的另外一种**模型压缩**的方式。

* 如果说“量化”是通过改变权重和激活值的表现形式从而让内存占用变小和计算变快的话
* “剪枝”则是直接“删除”掉模型中没有意义的，或者意义较小的权重，来让推理计算量减少的过程。
  * 更准确来说，是skip掉一些没有必要的计算
* 剪枝和量化是可以相辅相成的：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b172c0bc-71ca-42f1-b8f1-a30e2d20b71b)

* 同时模型剪枝也可以配合量化一起做
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/35a2ffad-038c-4087-ac82-885cf2853f79)




##### （2）模型剪枝的原因

为什么我们需要剪枝？主要是因为学习的过程中会产生**过参数化**导致会产生一些**意义并不是很大的权重**，或者**值为0的权重(ReLU)**。

* 对于这些权重所参与的计算是占用计算资源且没有作用的。
* 需要**想办法找到这些权重**并让硬件去skip掉这些权重所参与的计算
  * 找权重的方法可以以什么样的**粒度**来找这个0
    * per 权重？ per Channel ？
  * 找权重的方法可以以什么样的**型式**来归0
    * 规范化？随机？
  * 示意图
    * ![image-20240524115515701](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524115515701.png)
    * ![image-20240524115602871](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524115602871.png)



##### （3）模型剪枝的流程

* 1、获取一个已经训练好的初始模型
* 2、对这个模型进行剪枝
  * 我们可以通过训练的方式让DNN去学习哪些权重是可以**归零**的
    *  (e.g. 使用L1 regularization和BN中的scaling factor让权重归零)
    * ![image-20240524120355605](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524120355605.png)
  * 我们也可以通过自定义一些规则，手动的有规律的去让某些权重**归零**
    *  (e.g. 对一个1x4的vector进行2:4的weight prunning)
    * ![image-20240524120411854](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524120411854.png)
* 3、对剪枝后的模型进行fine-tuning
  * 有很大的可能性，在剪枝后初期的网络的精度掉点比较严重
  * 需要fine-tuning这个过程来恢复精度
  * Fine-tuning后的模型有可能会比之前的精度还要上涨
    * ![image-20240524120748553](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524120748553.png)
* 4、获取到一个压缩的模型
  * 其实如果到这个阶段对模型压缩还不够满足的话，可以回到step2循环
    * ![image-20240524120953330](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524120953330.png)



##### （4）模型剪枝的分类

1、模型剪枝可以按照剪枝的方法**按照一定规律与否**可以分为**<u>结构化剪枝</u>**，以及**<u>非结构化剪枝</u>**。

* 【1】**<u>结构化剪枝</u>**
  * 规定好每n个中删去x个，n和x固定好的
  * 或者以layer、channel为单位删除
* 【2】**<u>非结构化剪枝</u>**



2、同时，模型剪枝也可以按照剪枝的**粒度**与**强度**分为<u>**粗粒度剪枝**</u>，以及**<u>细粒度剪枝</u>**。

* 【1】<u>**粗粒度剪枝**</u>（Coarse Grain Pruning）
  * 从layer、channel层面剪枝
  * 这里面包括Channel/Kernel Pruning
    * Channel/Kernel Pruning是结构化减枝(Structured pruning)
    * 这个是比较常见的，也就是直接把某些卷积核给去除掉。
    * 比较常见的方法就是通过**L1Norm**寻找权重中影响度比较低的卷积核。
    * ![image-20240524121858227](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524121858227.png)
  * 优势和劣势
    * 优势
      * **不依赖于硬件**，可以在任何硬件（英伟达、高通.......）上跑并且得到性能的提升
    * 劣势
      * 由于减枝的粒度比较大(卷积核级别的)，所以有潜在的掉精度的风险
      * 不同DNN的层的影响程度是不一样的
      * **减枝之后有可能反而不适合硬件加速**(比如Tensor Core的使用条件是channel是8或者16的倍数)

* 【2】**<u>细粒度剪枝</u>**（Fine Grain Pruning）
  * 主要是对权重的各个元素本身进行分析减枝
  * 这里面可以分为**结构化减枝(structed)**与**非结构化减枝(unstructed)**
    * **<u>结构化减枝(structed)</u>**
      * Vector-wise的减枝: 将权重按照4x1的vector进行分组，每四个中减枝两个的方式减枝权重
        * ![image-20240524122944444](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524122944444.png)
      * Block-wise的减枝: 将权重按照2x2的block进行分区，block之间进行比较的方式来减枝block
        * ![image-20240524123005634](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524123005634.png)
    * <u>**非结构化减枝(unstructed)**</u>
      * Element-wise的减枝：每一个每一个减枝进行分析，看是不是影响度比较高
        * ![image-20240524123103397](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524123103397.png)
  * 优势和劣势：
    * 优势
      * 相比于Coarse Grain Pruning，精度的影响并不是很大
    * 劣势
      * 需要**特殊的硬件**的支持(Tensor Core可以支持sparse)
      * **需要用额外的memory**来存储哪些index是可以保留计算的
        * ![image-20240524123807415](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240524123807415.png)
      * memory的访问**不是很效率**(跳着访问)
      * 支持sparse计算的硬件内部会做一些针对sparse的tensor的**重编**，这个会比较耗时
        * 比如Tensor Core要做sparse的矩阵乘法，用索引选择哪些权重是可以跳过的，就涉及到weights和activation的重编



#### 4.4.2 Channel purning 算法与 L1-Norm 的关系

#### 4.4.3 Fine-grained structured sparse pruning

#### 4.4.4 分析 Sparse Tensor Core 硬件层面处理剪枝
