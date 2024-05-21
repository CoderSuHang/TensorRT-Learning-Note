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
  * ![image-20240521110127953](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521110127953.png)
  * 一个SM里面有：
    *  64个处理INT32的CUDA Core
    *  64个处理FP32的CUDA Core
    *  32个处理FP64的CUDA Core
    *  4个处理矩阵计算的的Tensor Core
  * 每一种精度在一个SM中的吞吐量（一个clk可以完成的计算数量）
    * ![image-20240521110305546](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521110305546.png)
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
      * ![image-20240521113133255](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521113133255.png)
    * 要完成4 x 8与 8x 4 的计算， 需要 8 * 16 = 128个clk才 可以完成
      * ![image-20240521113104914](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521113104914.png)
    * 当然，如果我们有16个 CUDA core的话，这些计 算并行，实际上是8个clk。 为了与Tensor Core比较， 这里只用一个CUDA Core
* Tensor Core ：
  * 使用一个Tensor Core 计算 C = A * B：
    * 第一代Tensor Core：
      * Tensor Core不是1个1个的 对FP16进行处理，而是4x4 个FP16一起处理，第一个clk先做A和B的前半段， 结果先存着
        * ![image-20240521113306509](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521113306509.png)
      * 第二个clk再处理A和B的后半 段，最后和前半段结果做个累 加，完成计算。所以说Tensor  Core处理4x8*8x4的计算只需 要1 + 1 = 2个clk
        * ![image-20240521113328684](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521113328684.png)
    * 第三代Tensor Core：
      * 可以1clk处理 4x8  * 8x8 的操作，也就是说1clk可以处理 256个FP16
        * ![image-20240521113610779](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240521113610779.png)
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



#### 4.1.3 FP32/FP16/INT8/INT4/FP8介绍



### 4.2 模型部署的几大误区

#### 4.2.1 FLOPS 并不能衡量模型性能

#### 4.2.2 TensorRT 并不能完全依靠

#### 4.2.3 CUDA Core 与 Tensor Core 的区别

#### 4.2.4 1x1 与 deptwise conv 的部署缺点



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
