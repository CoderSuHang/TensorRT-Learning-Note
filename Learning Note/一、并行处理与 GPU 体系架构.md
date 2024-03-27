# 深度学习模型部署TensorRT加速

## 一、并行处理与 GPU 体系架构

### 1.1 并行处理简介

#### 1.1.1 串行处理与并行处理的区别

* 串行处理：
  * 计算方法：多个任务或指令按照顺序依次执行，每个任务必须等待前一个任务完成后才能开始执行。
  * 处理单元：每个任务都是在同一个处理单元上顺序执行，因此无法同时进行多个任务的处理。
  * 应用场景：串行处理适用于简单的计算任务和单一任务的场景。
* 并行处理：
  * 计算方法：多个任务或指令同时执行，每个任务在不同的处理单元上独立运行。
  * 处理单元：多个处理单元可以同时处理多个数据，从而加快计算速度。
  * 应用场景：适用于数据密集型任务和需要同时处理多个任务的场景。
* 区别：
  * 执行方式：串行（多个任务按顺序依次执行，每个任务在同一个逻辑单元按顺序运行），并行（多个任务同时执行，每个任务在独立的处理单元上并行运行）。
  * 计算效率：并行效率高；
  * 适用场景：上述以介绍；
  * 资源需求：并行需求高。

#### 1.1.2 常见的并行处理

* 1、数据并行： 将数据分成多个部分，分配给不同的处理单元进行并行处理。每个处理单元使用相同的模型，但处理不同的数据片段。
* 2、任务并行：将计算任务分解成多个子任务，并分配给不同的处理单元并行执行。每个处理单元负责处理其中一个子任务，可以是相同或不同的模型。
* 3、模型并行：将大型模型拆分成多个部分，分配给不同的处理单元并行计算。每个处理单元负责计算模型的一个部分，然后将结果传递给其他处理单元进行进一步计算。
* 4、流水线并行：将计算流程分成多个阶段，并分配给不同的处理单元依次执行。每个处理单元在完成自己的任务后，将结果传递给下一个处理单元进行进一步计算。
* 5、并发并行：将不同的任务同时执行，而不必担心它们的执行顺序。并发并行可以在单个处理单元上实现多任务并行处理，通常通过多线程或多进程来实现。

### 1.2 GPU 并行处理

#### 1.2.1 GPU 与 CPU 的并行处理

* 1、GPU 并行处理：
  * （1）主要特点：高并行计算能力、高带宽内存、适用于数据并行、深度学习加速。
  * （2）并行处理和 GPU 体系架构之间的关系：
    * 并行处理：在并行处理中，可以利用多个处理单元（如多个CPU核心或GPU流处理器）同时执行多个指令，从而加快计算速度。
    * GPU 体系架构：具有大量的处理单元和高带宽内存，能够同时执行大量计算任务，适合处理数据密集型任务和并行计算。
      * GPU体系架构通常包含的关键组件：
        * **流处理器**：GPU中包含多个流处理器，也称为CUDA核心。每个流处理器负责执行计算任务，例如执行浮点运算和向量操作。
        * **多处理器**：GPU中的流处理器分组成多个多处理器，每个多处理器负责管理多个流处理器，并调度并行任务。
        * **全局内存**： GPU具有高带宽的全局内存，用于存储大规模的数据和模型参数。
        * **共享内存**： 共享内存是多个流处理器共享的高速缓存，用于加速多个流处理器之间的数据交换。
        * **纹理内存**： 纹理内存用于处理图像数据，适合对图像进行采样和滤波操作。
  * （3）GPU 并行处理的优势：
    * **并行计算能力**：GPU的并行结构使其能够同时执行多个计算任务，特别适用于数据密集型计算，如深度学习中的矩阵运算和卷积操作。
    * **高性能和吞吐量**：GPU的高带宽内存和多处理器架构使其能够提供更高的计算性能和数据吞吐量，加速大规模数据处理和模型训练。
    * **加速深度学习**：GPU广泛用于深度学习任务，如图像识别、目标检测和自然语言处理等，加速了模型训练和推理。
* 2、CPU 并行处理：
  * CPU具有多个核心和缓存，可以同时执行多个任务，但相比GPU，其并行计算能力较弱。
  * 主要特点：多核心处理、多线程处理、适用于任务并行、通用计算。
* 3、应用场景：
  * 在处理数据密集型任务和深度学习等需要高并行计算的场景下，GPU通常表现更优；
  * 而在通用计算和任务级并行处理的场景下，CPU则更为合适。
  * 很多情况下，GPU和CPU可以结合使用，充分发挥各自的优势，提高整体计算性能。

#### 1.2.2 Memory Latency

（1）Memory Latency（内存延迟）是指从发出内存请求到数据可供使用所需的时间。主要作用和影响如下：

* **1、系统性能**：内存延迟直接影响计算机系统的性能。如果内存延迟较高，CPU在等待数据时将会闲置，从而导致系统整体性能下降。
* **2、指令执行**： 在许多计算任务中，CPU需要频繁地从内存中读取数据和指令，如果内存延迟高，将会导致CPU等待数据的时间增加，从而降低指令执行速度。
* **3、缓存命中率**： 内存延迟直接影响缓存命中率。当CPU无法及时从内存中获取数据时，将会增加缓存未命中的可能性，从而导致CPU不得不从主内存中读取数据，进而增加内存延迟。
* **4、内存带宽利用率**： 当CPU等待数据时，内存带宽无法充分利用，导致内存带宽资源浪费。
* **5、计算性能**： 内存延迟对于计算任务的性能影响尤为重要。在数据密集型任务中，CPU需要频繁地访问内存，内存延迟会成为性能瓶颈。

（2）优化内存延迟对于提高计算机系统的性能至关重要。为了降低内存延迟，通常采取以下措施：

* **1、Cache 优化**：提高缓存的命中率，减少对主内存的访问，可以降低内存延迟。
* **2、预取技术**：提前将数据从内存加载到高速缓存，以减少后续访问时的延迟。
* **3、内存通道增加**：增加内存通道和带宽，提高内存读写速度。
* **4、内存层次结构优化** ：设计更优的内存层次结构，如高速缓存、本地存储器等，可以有效减少内存延迟。

#### 1.2.3 CPU 的优化方式

在 CPU 的并行处理方面，主要的优化能力包括以下几个方面：

* **多核心处理器**、**超线程技术**、**SIMD指令集**、**Cache优化**、**并行指令调度**、**并发执行**。

#### 1.2.4 GPU 特点

在GPU计算生态系统中，**CUDA、cuDNN和TensorRT**是三个重要的组件，它们各自具有不同的特点和作用。

（1）CUDA：CUDA是NVIDIA推出的**并行计算**平台和编程模型，它允许开发者在NVIDIA GPU上进行通用计算，并利用GPU的高并行计算能力加速各种计算任务。特点：**高并行计算能力**、**丰富的并行库**、**灵活性**。

（2）cuDNN：cuDNN是NVIDIA开发的深度学习计算库，基于CUDA平台。它为深度学习任务提供了高性能的基本操作，如卷积、池化、归一化等，以加速深度神经网络的训练和推理。特点：**优化的深度学习操作**、**兼容性**。

（3）**TensorRT**：TensorRT是NVIDIA开发的用于深度学习推理的高性能优化库。它能够自动优化深度学习模型，包括权重量化、卷积融合、内存优化等技术，以提高模型在GPU上的推理性能。特点：**推理性能优化**、**支持多种深度学习框架**、**支持多种精度**

**CUDA**提供了高性能的并行计算平台，**cuDNN**加速了深度学习任务的计算，**TensorRT**优化深度学习模型的推理性能，三者共同促进了GPU在高性能计算和深度学习领域的广泛应用。

### 1.3 环境配置

#### 1.3.1 版本选择：CUDA、cuDNN和TensorRT

##### 1.3.1.1 CUDA

由于资源有限不知道安装哪个版本，因此安装网上有教程的低版本11.0.2。

* 参考教程：
  * [CUDA安装教程（超详细）-CSDN博客](https://blog.csdn.net/m0_45447650/article/details/123704930)
* 安装记录：
  * 1.下载CUDA toolkit（toolkit就是指工具包）
    * [CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)
    * 这里选择11.8.0，原因是自己的jetson开发板cuda环境是11.4.0，但是该版本没有win11支持，所以看到以下文章：
      * [GPU版本的pytorch安装（显卡为3060ti，如何选择对应的cuda版本）_cuda版本怎么选-CSDN博客](https://blog.csdn.net/weixin_47250738/article/details/130170195?csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"130170195"%2C"source"%3A"sita1207"}&fromshare=blogdetail)
        * 根据文章介绍，我的笔记本是RTX4060，算力=8.9：
          * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/51369f7b-36c0-4976-afa0-f0accc0c9da6)

        * 对应版本为11.8：
          * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/46d278a5-f31a-46a1-b649-8a971823af87)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/67bfed92-0274-47cc-86bc-26454af3558d)

  * 2.这里没有安装 CUDA Samples，是因为11.8中没有相关安装选项，所以注册表也没有修改。
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/94102372-80fa-41dd-8f49-74a3a70fa1fe)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0247fefe-a5d2-4581-8f2e-feef5e2daebd)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d1912310-ec09-418e-acaf-65aa89358a82)

  * 3.按照教程检查一下注册表路径即可。

##### 1.3.1.2 cuDNN

* 安装教程：
  * 1.下载对应版本cuDNN（11.x最新）
    * [cuDNN Archive | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2712b0eb-b70e-47a8-a270-c5e861e490ff)

  * 2.解压复制相应文件夹，并且在系统变量path中添加四个路径即可。
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/dbb019c5-e22f-4fee-896d-0d81b3a5aa37)


##### 1.3.1.3 TensorRT

安装，极其复杂，需要搭建更复杂的环境：

参考教程：[【模型部署】TensorRT的安装与使用_tensorrt部署-CSDN博客](https://blog.csdn.net/qq_44747572/article/details/129022225?ops_request_misc={"request_id"%3A"168990146916800180635406"%2C"scm"%3A"20140713.130102334.."}&request_id=168990146916800180635406&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-129022225-null-null.142^v90^koosearch_v1,239^v3^control&utm_term=tensorrt安装&spm=1018.2226.3001.4187)

###### （1） cuda/cudnn以及虚拟环境的创建：

* 这里是非常困难的地方，因为是新电脑，Anaconda、Pycharm等软件都没有安装，所以需要逐个安装。
* 参考教程：[【环境配置】AI各种环境配置（anaconda、pycharm、cuda/cudnn、torch/torchvision等）_ai运行环境-CSDN博客](https://blog.csdn.net/qq_44747572/article/details/122453926?spm=1001.2014.3001.5502)
* 3.1.1 Anaconda环境配置：
  * 参考：
    * [【Anaconda教程01】怎么安装Anaconda3 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/75717350)
    * 【【手把手带你实战YOLOv5-入门篇】YOLOv5 环境安装】 https://www.bilibili.com/video/BV1G24y1G7qm/?share_source=copy_web&vd_source=a0dbe312acd17ef7f1fb082726d496a7
  * 本次选用与python3.8.10对应的版本（原因与Jetson开发板一致）
    * Anaconda版本与Python3版本对应关系：
      * [Anaconda版本与Python3版本对应关系 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/393803977)
    * 选择Anaconda3 2021.05：
      * [Anaconda超简单安装教程，超简洁！！！（Windows/Linux/Mac环境下，亲测有效） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/669733292)
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c7e3e01e-3fe2-41bb-8bac-007a451ead3f)

    * 其中有改动的是：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/424460ce-046b-45b5-895a-813677f54b1a)

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8a487582-fadb-450e-b5ef-dd819b7204ae)

      * 检查环境变量并没有添加成功，因此在这里添加以下4个环境变量：
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4c3b9a25-2296-474a-98b9-ab3879811697)

        * 教程中是5个，其中（E:\Anaconda\Library\usr\bin）没有找到对应路径，后续如果报错可以检查。
    * 添加清华镜像：
      * 参考：
        * 【保姆级Anaconda安装教程】 https://www.bilibili.com/video/BV1ns4y1T7AP/?share_source=copy_web&vd_source=a0dbe312acd17ef7f1fb082726d496a7
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b730078a-0c25-4005-a689-6e68b2915cc7)

* 3.1.2 Pycharm安装
  * 参考：
    * [【手把手带你实战YOLOv5-拓展篇】Pycharm基本使用与AutoDL服务器连接_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Ns4y1p7Ry/?spm_id_from=333.788&vd_source=0d02ed2f63507c727ce90624d9bd5e6a)
  * 选择下载版本：2022.1.3（社区版），原因和B站UP主（专业版）一致，他有yolo v5&8的教程，但是专业版可以远程，我不需要，所以用社区版
* 3.1.3 Anaconda和Pycharm环境搭建
  * Anacodna环境配置
    * 参考：
      * [【手把手带你实战YOLOv5-入门篇】YOLOv5 环境安装（重置版）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1bg4y1R7cs/?spm_id_from=333.788&vd_source=0d02ed2f63507c727ce90624d9bd5e6a)
      * [深度学习环境搭建详解（Anaconda、Pycharm、Cuda、Pytorch）-CSDN博客](https://blog.csdn.net/m0_73228309/article/details/136187809)
    * （1）查看环境列表：**conda env list**
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/530b1df5-13a2-460f-9725-8d3038a887a1)

    * （2）创建环境：**conda create -n env_name python=3.8**
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ef72d4b8-37af-43da-aa55-baaf50fc65a7)

    * （3）激活环境：**conda activate env_name**
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/68d6f739-0faa-4e63-8c1e-e0977cd6cf74)

    * （4）退出环境：**conda deactivate**
      * 退出当前环境后才能删除该环境。
    * （5）删除环境：**conda remove -n env_name --all**
  * Pycharm环境配置（Anaconda目录下）
    * [深度学习环境搭建详解（Anaconda、Pycharm、Cuda、Pytorch）-CSDN博客](https://blog.csdn.net/m0_73228309/article/details/136187809)
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/fdb297e8-5c81-451d-88bd-70756c274bb1)

    * 之后可以在Pycharm和Anaconda中验证是否创建成功：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/35eb4fe7-67ce-48cd-b9b1-d5ce5e7e303f)

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/59a82b1e-1958-43bf-90f9-765cdedcdc96)

* 3.1.4 Pytorch安装
  * 参考：
    * [【手把手带你实战YOLOv5-入门篇】YOLOv5 环境安装（重置版）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1bg4y1R7cs/?spm_id_from=333.788&vd_source=0d02ed2f63507c727ce90624d9bd5e6a)
    * [深度学习环境搭建详解（Anaconda、Pycharm、Cuda、Pytorch）-CSDN博客](https://blog.csdn.net/m0_73228309/article/details/136187809)
  * 这里版本选择11.8：
    * 在Anaconda中输入如下指令：
      * 激活Pytorch虚拟环境：
        * C:\Users\10482>conda activate Pytorch
      * 使用清华镜像源：
        * (Pytorch) C:\Users\10482>pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
      * 安装Pytorch11.8：
        * (Pytorch) C:\Users\10482>pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/319327ce-5b12-47a1-af34-3db5c81eddec)


* 3.2 根据 cuda 版本安装相对应版本的TensorRT

这里已经安装完成了CUDA 11.8、cuDNN 11.X:

* CUDA Toolkit 11.8
* cuDNN v8.9.7 for CUDA 11.x

现在安装TensorRT和zlibwapo.dll：

###### （2） TensorRT安装

 （a）安装包下载

* 网址：[NVIDIA TensorRT Download | NVIDIA Developer](https://developer.nvidia.com/tensorrt-download)
* 选择TensorRT 8.5 GA：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7292ed7c-d3d1-4f73-9bce-496107e8720c)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f89768e8-a172-470d-ac56-714ba2b3ecec)


（b）根据自己的python版本安装

* 查看python版本3.8.18：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9d7acdbb-35cf-4b3e-95d7-06e894550e3a)

* 安装对应版本的.whl文件：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b2c32ad7-661e-477d-8fe1-3d71e6ac175d)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8394ac9c-17f9-472d-b7bf-16ea25971ceb)

* 将【lib】文件夹下的动态链接库拷贝至CUDA安装位置的【bin】文件夹中：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6f66dcc3-9111-4161-a1c1-3d66c8d433e1)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a737a747-1461-4fe2-a864-6bdb3e085c0d)


###### （3）zlibwapi.dll安装

（安装cu116版本的pytorch可以避免此问题）

（a）安装包下载

* 网址：[ZLIB DLL Home Page (winimage.com)](http://www.winimage.com/zLibDll/)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8d522fd2-6098-4ce8-99f9-d2df1991a32f)


（b）解压将dll文件复制道CUDA的【bin】文件夹内：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c2dbe355-1f4f-45ee-b229-abeeb3e9c70f)

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b56af18b-0b0a-4419-9be2-0bb031fe811a)

#### 1.3.2 VMware虚拟机安装Ubuntu22.04.2

##### 1.3.2.1 VMware

（1）安装网址：

* [VMware - Delivering a Digital Foundation For Businesses](https://www.vmware.com/)

（2）下载安装：

* 参考CSDN：
  * [安装虚拟机（VMware）保姆级教程（附安装包）_vmware虚拟机-CSDN博客](https://blog.csdn.net/weixin_74195551/article/details/127288338?ops_request_misc=%7B%22request%5Fid%22%3A%22171149993816800227423899%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=171149993816800227423899&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-127288338-null-null.142^v100^pc_search_result_base5&utm_term=vmware虚拟机安装教程&spm=1018.2226.3001.4187)
* 参考B站：
  * [两分半钟完成VMware安装及Linux-Ubuntu安装（全程无废话）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1W34y1k7ge/?spm_id_from=333.337.search-card.all.click&vd_source=0d02ed2f63507c727ce90624d9bd5e6a)

##### 1.3.2.2 Ubuntu

（1）下载网址：

* http://mirrors.zju.edu.cn/ubuntu-releases/20.04/

（2）安装20.04
