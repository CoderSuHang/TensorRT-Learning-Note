# CUDA与TensorRT部署

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

### 1.3 环境配置new

本次选择使用 Windows terminal + wsl2 来运行 Ubuntu 系统，使用 vscode作为编辑器， 服务器选择 Ubuntu22.04 + docker。

#### 1.3.1 WSL2安装

参考视频教程：

【Windows11安装WSL2】 https://www.bilibili.com/video/BV1n14y1x7Y7/?share_source=copy_web&vd_source=a0dbe312acd17ef7f1fb082726d496a7

##### （1）搜索功能勾选【适用于Linux的Windows子系统】和【虚拟机平台】

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6b3b5294-730d-4f5a-afcf-a1075e3ea75b)

##### （2）点击立即重新启动

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/77197e48-8e29-433e-94e1-e63c50ceff12)


##### （3）安装Linux发行版，打开微软商店，安装Ubuntu 22.04

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8e29f278-bdc0-4b6a-add8-4f53732414ba)

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/940b5072-7ec8-4762-8e62-d3fb4cb10b49)


##### （4）报错：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a02fdc9f-1a56-42d1-82e6-dd5418988497)

* 使用powershell进行更新：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2369a1a6-d430-4053-a448-242a0ccf4c88)


##### （5）安装Terminal

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/89058a3a-4637-4abf-a031-f35e0d1b1ac2)


##### （6）启动Ubuntu

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9faa16cb-3d1b-4b09-aeb9-3e1a401ab881)


##### （7）其他

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/024c6cde-675a-4414-bad6-fc278a35610f)


##### （8）WSL2安装位置迁移，因为原文件安装在C盘，导致C盘空间不足

* ```c++
  C:\Users\10482\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_79rhkp1fndgsc\LocalState
  ```

* 参考连接：

  * [WSL2安装 迁移 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/593297123)

#### 1.3.1 版本选择：CUDA、cuDNN和TensorRT 

##### （1）查看本机版本：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0dcafa00-7a31-44ac-8128-e4b8163f6066)

* Ubuntu：22.04.3
* Driver Version：551.86
* CUDA Version：12.4（是能够兼容的最大CUDA版本）

##### （2）TensorRT 8.5.1 安装下载

* 下载连接：

  * [Installation Guide :: NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-851/install-guide/index.html#downloading)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/931cb207-923f-43c2-a059-aa01f7505572)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2374e480-bc35-4651-bbd7-10fe0a1ce8ac)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e63daeaa-b211-4577-9851-8f336c453491)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e400f54d-8e8a-4e6d-8db0-fcdc571bf28a)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/50663b63-2b38-48df-abd1-93e288c5ffbe)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/91bada89-f520-4e90-b2fa-f4881f7b15a8)

* 配置环境变量：

  * 输入【vim ~/.bashrc】打开vim编辑器：

    * ```python
      suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ vim ~/.bashrc
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/acdba0f3-c75a-4b28-86ba-43101d4afd08)


  * 在文件最后一行加入下面指令：

    * ```python
      export PATH=/mnt/e/Software/LinuxOS/wsl2/packages/TensorRT-8.5.1.7/bin:$PATH
      export LD_LIBRARY_PATH=/mnt/e/Software/LinuxOS/wsl2/packages/TensorRT-8.5.1.7/lib:$LD_LIBRARY_PATH
      export LIBRARY_PATH=/mnt/e/Software/LinuxOS/wsl2/packages/TensorRT-8.5.1.7/lib:$LIBRARY_PATH
      ```

    * 按下【ESC】输入【:w】保存，并输入【:q】退出

    * 在终端输入【source ~/.bashrc】，保存后刷新环境变量。

      * ```python
        suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ source ~/.bashrc
        ```

* 运行检查，输入【trtexec】

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5758142e-91f2-4841-8974-f63fe4b156d8)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/acd72465-d875-4629-a71e-c009a1a6071d)


* 运行一个例程试试：

  * ```python
    suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages/TensorRT-8.5.1.7/samples/sampleOnnxMNIST$ make -j16
    ```

  * 报错：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b429fb3e-5570-4e6c-819d-cc3ce22b0f77)


    * 原因是没有安装g++，运行安装指令即可：

      * ```python
        sudo apt-get install g++
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a146262c-88d6-4fb8-9bf1-91df9636c539)


  * 重新运行：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f113cd3c-d1a9-4f32-8d88-26a52349d625)


    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9e22d507-7e92-44a5-ae76-5e2be48ee83a)


    * 报错：

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9eb3ee8f-25dd-4bcc-8b29-73ec43870970)


      * ```c++
        Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory
        ```

      * 这是因为没有加入cuDNN路径，在bashrc文件中添加即可：

        * ```python
          export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64/stubs/:/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7/cudnn/lib:$LD_LIBRARY_PATH
          ```

        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/095a421b-eb9b-4d42-b52e-ae5f3e2c174a)


    * 再次运行：

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/699cb68a-f5db-4818-9e1e-88b2e64fb5eb)

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f95d1d11-b55e-4178-96e7-67b5a0f771bf)


##### （3）CUDA11.7 安装下载

* 下载连接：

  * [Release Notes :: NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html#rel-8-5-1)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6d7a168f-5a8b-43c2-8553-dee6885ecb3c)


  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7c253dc9-2d5a-42ad-b49f-e503bbb85634)


  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d8860a45-e52f-4732-aefd-3f6b7bd2125d)


  * 运行指令：

    * ```python
      wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
      ```

    * 首先要在Ubuntu中新建文件夹：

      * ```python
        suhang@Y9000P:/mnt/c/Users/10482$ mkdir packages
        ```

    * 打开文件夹：

      * ```python
        suhang@Y9000P:/mnt/c/Users/10482$ cd packages
        ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b77c9e7d-a18a-4525-82c6-624e83960272)

    * 切换下载位置：

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/14fefaf9-6ac2-4d58-abdc-b9f8d70314f4)


* 运行安装包

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bc1a7513-c07b-4318-b1ff-6e6a4c0e351f)


  * 报错，用--override进行忽略：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8398a0d2-a9fb-4227-ba86-feb22ba2d8fc)


  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/28bc3665-bc74-475d-b10a-b097879de48d)


  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1ea2bd4a-0525-45c0-8928-3e74c71513cd)


    * 这表示 NVIDIA Driver 和 CUDA Toolkit 已安装完毕。后半段安装信息提示我们修改[环境变量](https://link.zhihu.com/?target=https%3A//blog.csdn.net/dlutbrucezhang/article/details/8811456) PATH 和 LD_LIBRARY_PATH. 在 ~/.bashrc 文件中写入

      * 注意当使用如下指令时，添加的路径成为临时路径，如果关闭终端再次打开，会无法启动CUDA：

        * ```python
          suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ export PATH=$PATH:/usr/local/cuda-11.7/bin
          suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
          ```

      * 解决办法：

        * 输入【vim ~/.bashrc】打开vim编辑器：

          * ```python
            suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ vim ~/.bashrc
            ```

          * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3813252d-7962-460d-9a3d-1ee898b2b1de)


        * 在文件最后一行加入下面指令：

          * ```python
            export PATH=/usr/local/cuda-11.7/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
            ```

          * 按下【ESC】输入【:w】保存，并输入【:q】退出

          * 在终端输入【source ~/.bashrc】，保存后刷新环境变量。

            * ```python
              suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ source ~/.bashrc
              ```

* 查看安装版本：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/af305e07-3c36-4e5b-ae08-0d818dd95ea1)


##### （4）cuDNN 8.6.0 安装下载

* 下载网址：
  * [cuDNN Archive | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9075c3b1-16ed-4abb-b17a-1f1ddf1d9e54)

* 解压所有压缩包：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d8702034-f081-4790-a657-d5563abd0007)


* 在当前目录下运行如下指令

  * ```python
    sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
    sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/fcc0c4a9-c08d-446b-9db1-0ceb950c9bb1)


* 检查

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/02de7c51-55b4-465b-933d-abb936a16d7c)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/16febd65-42f0-4f9b-9ccb-4646b98583da)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ea8eb5d0-4e28-4186-9738-351c3c544812)


#### 1.3.3 fish shell安装

是一款功能强大的命令行工具，它具有自动补全、语法高亮和更友好的用户界面等特性。

##### （1）参考教程

* [如何在 Ubuntu 中安装和配置 Fish Shell？(1) - 芒果文档 (imangodoc.com)](https://imangodoc.com/NFwcPLBy.html)

##### （2）安装 Fish Shell

* ```python
  sudo apt install fish
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/48791055-27a8-48c4-9989-4c61d0e68490)


* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/06b6af61-5fea-48ee-9e37-d65d0b971c52)

##### （3）运行fish

* ```python
  suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ fish
  ```

* fish的其他用法：

  * [Fish Shell 入门体验 - Eslzzyl - 博客园 (cnblogs.com)](https://www.cnblogs.com/eslzzyl/p/16902538.html)

##### （4）安装fisher包

* 参考教程
  * [小知识点系列：终端下如何安装 Fisher_fish插件-CSDN博客](https://blog.csdn.net/puchunwei/article/details/131295476)

* ```python
  suhang@Y9000P /m/e/S/L/w/packages> curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/98746c5a-2898-4c9f-85e1-4cc9e29e17d0)


（5）快捷跳转指令Z

* 需要在实体机上安装，这里有教程，以后有机会尝试：
  * [【干货】你不知道的 Linux 命令使用技巧_linux快捷跳转类似mac的z命令-CSDN博客](https://blog.csdn.net/XMWS_IT/article/details/119108005)

#### 1.3.4 exa使用

exa可以使让终端显示更具体的文件夹列表，有助于我们查找文件

##### （1）参考教程

* [exa · a modern replacement for ls](https://the.exa.website/)

##### （2）安装exa

* ```python
  suhang@Y9000P /m/e/S/L/w/packages [100]> sudo apt install exa
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8493395f-c26e-462d-bb22-645d11f0713a)


##### （3）使用exa

* ```python
  suhang@Y9000P /m/e/S/L/w/packages> exa -l
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/830bbdc4-d0a5-4041-bbd1-970d1d4a29f7)


* 可以创建快捷方式ll，但是我无法打开fish内部文件，以后有办法再打开吧，参考视频1.3.3 中的05:10的内容。这个可能需要一个Ubuntu实体机才行。wsl我能root但是找不到文件。

  * ```python
    suhang@Y9000P /m/e/S/L/w/packages> vim ~/.config/fish/
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/723aed7c-0f2b-4175-8c04-980a096adb22)


#### 1.3.5 tmux使用

安装tmux进行多window多session的管理

##### （1）安装教程

* [Linux安装tmux_linux 安装tmux-CSDN博客](https://blog.csdn.net/qq_35985044/article/details/131820250)

##### （2）运行在线安装指令

* ```python
  sudo apt-get install tmux
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bb35dc2b-5e50-45fd-bed0-ab56f4435af2)


#### 1.3.6 netron使用

安装netron进行DNN网络架构图的分析

##### （1）运行在线安装指令

* ```python
  suhang@Y9000P /m/e/S/L/w/packages> pip3 install netron
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/afc5395d-7ad4-4a70-af26-09d7325f9113)


### 1.4 服务器环境配置




#### 1.4.1 安装NVIDIA Container Toolkit

##### （1）参考教程

* [sudo apt-get install -y nvidia-container-toolkit-base执行报错解决方案_unable to locate package nvidia-container-toolkit--CSDN博客](https://blog.csdn.net/qq_28593347/article/details/131471058)

##### （2）运行指令安装

* 配置apt源

  * ```python
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
    
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

* 安装nvidia-container-toolkit

  * ```python
    sudo apt-get update
    
    sudo apt-get install -y nvidia-container-toolkit-base
    ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0b73aaaf-63ed-4569-ba41-148e7abb32f3)


##### （3）根据官网指令检查

* ```python
  suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ nvidia-ctk --version
  
  suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ sudo nvidia-ctk cdi generate --output=/etc/cdi/nvdia.yaml
  
  suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ grep "  name:" /etc/cdi/nvdia.yaml
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/26d44425-a818-4551-8114-585aa01394ed)


* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ee6835d8-0360-4c14-9be9-834f7d42bae3)


#### 1.4.2 安装Docker

##### （1）参考链接

* [ubuntu安装nvidia-docker_ubuntu_icodekang-华为云开发者联盟 (csdn.net)](https://huaweicloud.csdn.net/638db459dacf622b8df8c9eb.html)

##### （2）安装docker

* ```python
  curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ecab39e1-7581-450f-9592-537d8a455606)


##### （3）设置源

* ```python
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
     && curl -fsSL https://nvidia.github.io/libnvidia-docker/gpgkey | sudo apt-key add - \
     && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/762d1872-61fb-400a-a468-195d37808d48)


* 安装失败，待解决：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2a32182b-b6b3-4968-a869-ab4b5f70208f)


  * Warning: apt-key is deprecated. Manage keyring files in trusted.gpg.d instead (see apt-key(8)).

  * ```python
    W: https://mirrors.tuna.tsinghua.edu.cn/ubuntu/dists/bionic/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
    W: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
    W: https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
    W: https://nvidia.github.io/nvidia-container-runtime/stable/ubuntu18.04/amd64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
    W: https://nvidia.github.io/nvidia-docker/ubuntu18.04/amd64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
    ```

  * 问题未解决，但影响后续

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/48ee61a5-0181-41cc-9527-3cc4628f8e74)

    * ```python
      suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$ curl https://get.docker.com | sh \
        && sudo systemctl --now enable docker
        % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                       Dload  Upload   Total   Spent    Left  Speed
      100 21927  100 21927    0     0  21449      0  0:00:01  0:00:01 --:--:-- 21454
      # Executing docker install script, commit: e5543d473431b782227f8908005543bb4389b8de
      
      WSL DETECTED: We recommend using Docker Desktop for Windows.
      Please get Docker Desktop from https://www.docker.com/products/docker-desktop/
      
      
      You may press Ctrl+C now to abort this script.
      + sleep 20
      + sudo -E sh -c apt-get update -qq >/dev/null
      [sudo] password for suhang:
      W: https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
      W: https://nvidia.github.io/nvidia-container-runtime/stable/ubuntu18.04/amd64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
      W: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
      W: https://nvidia.github.io/nvidia-docker/ubuntu18.04/amd64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
      + sudo -E sh -c DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apt-transport-https ca-certificates curl >/dev/null
      + sudo -E sh -c install -m 0755 -d /etc/apt/keyrings
      + sudo -E sh -c curl -fsSL "https://download.docker.com/linux/ubuntu/gpg" | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
      + sudo -E sh -c chmod a+r /etc/apt/keyrings/docker.gpg
      + sudo -E sh -c echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" > /etc/apt/sources.list.d/docker.list
      + sudo -E sh -c apt-get update -qq >/dev/null
      W: https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
      W: https://nvidia.github.io/nvidia-container-runtime/stable/ubuntu18.04/amd64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
      W: https://nvidia.github.io/nvidia-docker/ubuntu18.04/amd64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
      W: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
      + sudo -E sh -c DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-ce-rootless-extras docker-buildx-plugin >/dev/null
      E: Failed to fetch https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/amd64/containerd.io_1.6.28-2_amd64.deb  Connection timed out [IP: 13.224.163.81 443]
      E: Failed to fetch https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/amd64/docker-buildx-plugin_0.13.1-1%7eubuntu.22.04%7ejammy_amd64.deb  Connection timed out [IP: 13.224.163.81 443]
      E: Failed to fetch https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/amd64/docker-ce_26.0.0-1%7eubuntu.22.04%7ejammy_amd64.deb  Connection timed out [IP: 13.224.163.23 443]
      E: Failed to fetch https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/amd64/docker-ce-rootless-extras_26.0.0-1%7eubuntu.22.04%7ejammy_amd64.deb  Connection timed out [IP: 13.224.163.23 443]
      E: Failed to fetch https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/amd64/docker-compose-plugin_2.25.0-1%7eubuntu.22.04%7ejammy_amd64.deb  Connection timed out [IP: 13.224.163.23 443]
      E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
      suhang@Y9000P:/mnt/e/Software/LinuxOS/wsl2/packages$
      ```

##### （4）重复一下官网教程

* [Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit 1.14.5 documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)
* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3ca0b6d5-f890-4e69-b9db-41f02f2d9ba8)


##### （5）验证

我们可以启动一个container查看nvidia-smi的运行状况：

* ```python
  sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
  ```

* 报错：

  * ```python
    Failed to initialize NVML: GPU access blocked by the operating system
    Failed to properly shut down NVML: GPU access blocked by the operating system
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0a42fd28-818c-4559-8dab-2fe3758c8c54)


* 解决办法：把rootless模式下的配置全部无效掉

  * 参考博客：

    * [docker run时出现的Failed to initialize NVML: GPU access blocked by the operating system问题-CSDN博客](https://blog.csdn.net/weixin_46146538/article/details/136368447)

  * ```python
    sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups=false --in-place
    rm $HOME/.config/docker/daemon.json
    sudo systemctl --user restart docker
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/fdd32060-1ce1-4b3d-8cd8-c17472501924)


##### （6）使用docker images报错

* ```python
  docker images
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/db53eec8-952e-44b6-9368-ff24ec33b8ff)


* 在指令前加入sudo来执行

##### （7）其他指令

* 【--rm】如果我们在验证指令那里没有用到"--rm"，这时候就会在本地docker里创建当前镜像，并不会删除：

  * ```python
    sudo docker run --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
    ```

  * 这是便可以用下面指令查看存在的镜像容器：

    * ```python
      sudo docker ps -a
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/37a7e87b-d48f-4704-951f-5a4693900246)


  * 使用下面指令可以删除这个镜像容器（记得输入目标容器的ID前四位）：

    * ```python
      sudo docker rm 0601
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2f71a39d-382b-4f2a-96d9-c08cf56e4715)


##### （8）免去【sudo】指令执行

* 可以将docker加入用户群组当中，但是我没成功：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7e750061-6c43-4871-a537-ea74c0ceee05)


#### 1.4.3 从NVIDIA NGC中的官方release note寻找对应版本

创建镜像的时候我们需要通过TAG来创建，但是TAG是否满足主机环境需要在官方文档中确定，并根据自己显卡驱动要求的CUDA最大版本，以及需要运行的模型所需要环境来选择。

##### （1）官方网址

* 官方文档介绍页面
  * [:: (nvidia.com)](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/81d2abb2-3c5f-49b5-b6e1-934715a16960)

* TAG文件提供页面（创建镜像需要根据它来创建）：
  * [:: (nvidia.com)](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/fe6ebbeb-9838-4c8f-b39c-69d78a1f8b90)


##### （2）查看环境所需配置

* BEVFusion：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ab8fde9b-8599-4276-857b-2bdd8c759308)

* Host：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c5d28b59-252d-44aa-829e-4c2c1666afc2)

* 这里我们选择【TensorRT Release 22.08】，虽然BEVFusion环境要求TensorRT版本要大于8.5.0，但是我们可以在创建容器的时候，将路径变换到指定目录下，实现TAG的运行
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9722af93-ad2c-40eb-b79a-172a5e30497a)


##### （3）下载TAG

* 拷贝镜像：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0480cfc0-ec10-4939-a7c1-31be9ba891a1)


* ```python
  nvcr.io/nvidia/tensorrt:22.08-py3
  ```

* 等待创建dockerfile时候使用

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/981bb771-c5ee-4638-8743-d246a5f117e7)


#### 1.4.4 根据需求自行创建dockerfile

##### （1）导入TAG镜像

* 选择你要使用的TAG

* ```python
  # For more detail, check NGC
  FROM nvcr.io/nvidia/tensorrt:22.08-py3
  ```

##### （2）设置时区

* 设置自己的时区，注意要和自己所在时区一致

* ```python
  ENV TZ=Asia/Beijing
  ARG user=suhang
  
  # Set timezone in case of interation during installation
  RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
  ```

##### （3）配置opencv

* 安装opencv需要的依赖包：

  * ```python
    # install packages for opencv
    RUN apt-get update
    RUN apt install -y --no-install-recommends \
      build-essential cmake git pkg-config libgtk-3-dev \
      libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
      libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev \
      python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev \
      build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev \
      libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
      libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev \
      python3-numpy libtbb2 libtbb-dev libdc1394-22-dev 
    RUN apt-get -y clean && rm -rf /var/lib/apt/lists/*
    ```

* 创建opencv，可以在opencv官网中查看

  * ```python
    #Build opencv4.5.5
    RUN mkdir /root/opencv_build
    WORKDIR /root/opencv_build/
    RUN git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git && \
      cd /root/opencv_build/opencv && git checkout 4.5.5 && \
      cd /root/opencv_build/opencv_contrib && git checkout 4.5.5 && \
      cd /root/opencv_build/opencv && mkdir build
    
    WORKDIR /root/opencv_build/opencv/build
    RUN cmake -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python2=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
      -D CMAKE_INSTALL_PREFIX=/usr/local ..
    RUN make -j16 && make install
    WORKDIR /root
    RUN rm -rf opencv_build
    ```

##### （4）安装推理所需安装包

* 安装和配置：

  * ```python
    # install packages that are used for trt inferences
    RUN apt-get update && apt install -y --no-install-recommends \
      sudo libgflags-dev libboost-dev libxml2-dev \
      libyaml-cpp-dev sqlite3 libsqlite3-dev libboost-all-dev \
      fish lsb-release peco feh fim openssh-server tmux curl
    RUN apt-get -y clean && rm -rf /var/lib/apt/lists/*
    
    # install netron to show model structure
    RUN pip install netron
    
    # set up fish shell and exa
    RUN curl -Lo exa.zip "https://github.com/ogham/exa/releases/latest/download/exa-linux-x86_64-v0.10.1.zip" && \
      sudo unzip -q exa.zip bin/exa -d /usr/local && rm exa.zip
    ```

##### （5）创建用户组

* 将系统默认创建到root目录进行修改：

  * ```python
    # set userinfo
    RUN useradd -rm -c ${user} -u 1000 -d /home/${user} -s /bin/bash -G sudo ${user}
    SHELL ["/bin/bash", "-o", "pipefail", "-c"]
    RUN echo "$user   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    
    RUN apt-get update
    ```

##### （6）拷贝同级目录中的文件夹到容器当中去

* ```python
  # copy dotfiles
  COPY my_dot_files/tmux/* /home/${user}/
  COPY my_dot_files/fish/ /home/${user}/.config/fish
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/65ebda22-423a-4734-b589-6bdbdbb9873d)


##### （7）配置工作目录

* ```python
  # set working directory of trt webinar
  RUN mkdir -p /home/${user}/workspace/
  RUN chown -R ${user}:users /home/${user}
  ```

##### （7）配置openssh-server

* ```python
  # ssh setting
  RUN systemctl enable ssh
  ```

##### （8）更改默认启动目录，启动用户

* ```python
  # set default working directory and user
  WORKDIR /home/${user}
  USER ${user}
  ```

#### 1.4.6 运行docker

##### （1）创建image

使用脚本创建docker，在当前目录下的dockerfile创建image：

* 脚本在【scripts】文件夹下：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d7917919-e28c-4532-ae5a-ea308a6d2a5c)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/686bd873-f38b-4e5b-ab14-16c3dac690ab)


运行脚本创建image指令：

* ```python
  sudo bash scripts/build-docker.sh v1.x
  ```

  * v1.x自己设定
  * 等待后台创建完成。

* 报错：

  * ```python
    ERROR: failed to solve: process "/bin/sh -c curl -Lo exa.zip \"https://github.com/ogham/exa/releases/latest/download/exa-linux-x86_64-v0.10.1.zip\" &&   sudo unzip -q exa.zip bin/exa -d /usr/local && rm exa.zip" did not complete successfully: exit code: 28
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f3bd60c1-d0c0-47f7-b7ee-554c6ab6ff81)


* 原因是没法成功下载exa的安装包，按照这个网址跟踪了一下，发现安装包的网址已经更新，因此在dockerfile中安装exa的指令网址更新即可：

  * 新的网址：
    * https://github.com/ogham/exa/releases/download/v0.10.1/exa-linux-x86_64-v0.10.1.zip
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a247f764-c2f4-43bb-bef2-dbc70da57bb2)


运行完毕：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/460bd426-0597-4e55-b52e-294801b22c55)


查看创建完成的image：

* ``` python
  sudo docker images
  ```

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f1d04e1d-7593-4b72-8631-3c761583235d)


##### （2）创建容器

使用脚本根据创建好的image去创建一个容器

* 脚本在【scripts】文件夹下：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a9344bed-4ed3-4da7-895a-c09f84964b29)


  * ```python
    #!/bin/sh/
    
    docker run -it \
    	--name trt_starter_${1} \
      --gpus all \
    	-v /tmp/.X11-unix:/tmp/.X11-unix \
    	-v /home/suhang/Code:/home/suhang/Code \
      -p 8090:22 \
    	-e DISPLAY=:1 \
    	trt_starter:cuda11.4-cudnn8-tensorrt8.2_${1} \
      fish
    ```

    * 其中：
      * 【docker run -it】让用户能够和dockers container交互
      * 【--name trt_starter_${1}】指定一个名字
      * 【--gpus all】让你能够访问到主机端的GPU
      * 【-v /tmp/.X11-unix:/tmp/.X11-unix】让你能从终端打开容器container里的gui操作
      * 【-v /home/suhang/Code:/home/suhang/Code】是将主机端文件夹与容器端文件夹绑定
      * 【-p 8090:22】开放端口，能直接ssh进入容器中
      * 【-e DISPLAY=:1】
      * 【trt_starter:cuda11.4-cudnn8-tensorrt8.2_${1}】对应的docker容器的镜像版本

* 运行脚本创建指令：

  * ```python
    sudo bash scripts/run-docker.sh v1.2
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5dbadef7-5f08-4c1b-bf7a-ca599eb9296c)


  * 报错：

    * NVIDIA驱动不匹配：

      * ```python
        suhang@Y9000P /m/e/S/L/w/t/c/1.0-build-environment> sudo bash scripts/run-docker.sh v1.2
        
        =====================
        == NVIDIA TensorRT ==
        =====================
        
        NVIDIA Release 22.08 (build 42105201)
        NVIDIA TensorRT Version 8.4.2
        Copyright (c) 2016-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        
        Container image Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
        
        https://developer.nvidia.com/tensorrt
        
        Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
        
        This container image and its contents are governed by the NVIDIA Deep Learning Container License.
        By pulling and using the container, you accept the terms and conditions of this license:
        https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
        
        To install Python sample dependencies, run /opt/tensorrt/python/python_setup.sh
        
        To install the open-source samples corresponding to this TensorRT release version
        run /opt/tensorrt/install_opensource.sh.  To build the open source parsers,
        plugins, and samples for current top-of-tree on master or a different branch,
        run /opt/tensorrt/install_opensource.sh -b <branch>
        See https://github.com/NVIDIA/TensorRT for more information.
        WARNING: Detected NVIDIA NVIDIA GeForce RTX 4060 Laptop GPU GPU, which is not yet supported in this version of the container
        ERROR: No supported GPU(s) detected to run this container
        
        ~/.config/fish/config.fish (line 92):
        conda activate trt-starter
        ^
        from sourcing file ~/.config/fish/config.fish
                called during startup
        welcome back!!
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/46135723-fb03-43d9-8b1c-82860c552560)


        * 解决办法：可能是路径设置问题，在这里先不管了，继续看看再说。

  * 报错：重启之后再次run这个docker会说你已经有容器使用这个了。必须删除（或重命名）该容器才能重用该名称。

    * ```python
      suhang@Y9000P /m/e/S/L/w/t/c/1.0-build-environment [125]> sudo bash scripts/run-docker.sh v1.2
      docker: Error response from daemon: Conflict. The container name "/trt_starter_v1.2" is already in use by container "16393009b5733f4b9e4de4cc8daecb019e7c798c78764dc85badfa24d5682b43". You have to remove (or rename) that container to be able to reuse that name.
      See 'docker run --help'.
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2354b979-0b47-4d07-ac2a-037c87bedd47)


    * 解决办法：

      * 查看运行的容器：

        * ```python
          suhang@Y9000P /m/e/S/L/w/t/c/1.0-build-environment [125]> sudo docker ps -a
          CONTAINER ID   IMAGE                                          COMMAND                  CREATED          STATUS                     PORTS     NAMES
          16393009b573   trt_starter:cuda11.4-cudnn8-tensorrt8.2_v1.2   "/opt/nvidia/nvidia_…"   19 minutes ago   Exited (0) 9 minutes ago             trt_starter_v1.2
          ```

        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6c331c93-18de-4449-b122-fc8a83fd6610)


      * 删除该容器：

        * ```python
          suhang@Y9000P /m/e/S/L/w/t/c/1.0-build-environment> sudo docker rm 1639
          1639
          suhang@Y9000P /m/e/S/L/w/t/c/1.0-build-environment> sudo docker ps -a
          CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
          ```

        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b2791881-3bfa-467f-b27d-94170c0616f0)


（3）运行nvidia-smi确认可执行：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0d6ab0fb-0cf3-4974-a91f-bdcac20d9af5)


### 1.5 编辑器环境配置

这个博主记录的很好：

* [配置VScode开发环境-CUDA编程_vscode cuda-CSDN博客](https://blog.csdn.net/qq_45032341/article/details/133843192)

#### 1.5.1 创建compile_commands.json

##### （1）第一次可以安装一个Bear插件

* Bear是一个自动创建compile_commands.json的插件

* 官方网址：

  * [rizsotto/Bear: Bear is a tool that generates a compilation database for clang tooling. (github.com)](https://github.com/rizsotto/Bear)

* 安装过程：

  * 在终端中输入

    * ```python
      sudo apt-get install bear
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bac6fd31-23a6-41a2-88b6-f1361eebecfd)


* 使用过程：

  * 输入

    * ```python
      bear -- make -j16
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c55cc46b-ecf4-4b8f-a6c8-fa3c3bd0259c)


    * 报错：找不到【cuda_runtime.h】

      * ```
        src/utils.hpp:4:10: fatal error: cuda_runtime.h: No such file or directory
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/95b6843a-f0f1-4c59-b162-80e20b3e090d)


      * 解决办法：

        * 加入新的路径：
          * [【已解决】 fatal error: cuda_runtime.h: 没有那个文件或目录_fatal error: cuda_runtime_api.h: 没有那个文件或目录-CSDN博客](https://blog.csdn.net/weixin_45617478/article/details/116209903)
        * 并且指定cuda版本：
          * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0b3111a9-9165-4275-821b-b3ef7680fba7)


    * 但是有有了新的问题：

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d34d7d80-1140-41e7-a0cd-f3e5f268065b)



      * 这个问题说找不到Makefile.config文件，问了ChatGPT说是因为在Makefile中指定了路径位置，但是没有在当前文件位置的前两级目录中找到：

        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/10c81078-a5f0-42d3-a58a-e8bab489fc50)

        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8c2f39d7-1876-45dc-96f2-4cb027f14c37)


      * 我按照ChatGPT提供的建议修改了Makfile，虽然问题能够解决，但是本质问题依旧存在：

        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e7eb57a0-087c-40d5-97ce-efeac07ce78e)

        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3ac5401a-50a3-4889-940a-28a05cbba2c1)


      * 因此我又重新看了下博主的工程文档，发现博主有说明需要git他的工程，进而配置config文件：

        * [kalfazed/tensorrt_starter：这个存储库提供了从头开始学习 CUDA 和 TensorRT 的指南。 (github.com)](https://github.com/kalfazed/tensorrt_starter?tab=readme-ov-file#chapter2-cuda-programming)

        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/47c7745f-9e70-4879-8202-b3c9da6a8a63)


          * git的时候报错可以看这个文档解决：

            * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/46f4861a-a03f-4bcc-85e3-304ec8e776f7)


            * 非公开内容：

              * ```python
                suhang@Y9000P /m/e/S/L/wsl2> git clone git@github.com:kalfazed/tensorrt_starter.git
                Cloning into 'tensorrt_starter'...
                The authenticity of host 'github.com (20.205.243.166)' can't be established.
                ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
                This key is not known by any other names
                Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
                Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.
                git@github.com: Permission denied (publickey).
                fatal: Could not read from remote repository.
                
                Please make sure you have the correct access rights
                and the repository exists.
                suhang@Y9000P /m/e/S/L/wsl2 [128]> git config --global user.name "CoderSuHang"
                suhang@Y9000P /m/e/S/L/wsl2> git config --global user.email "1048227620@qq.com"
                suhang@Y9000P /m/e/S/L/wsl2> git config --global credential.helper store
                suhang@Y9000P /m/e/S/L/wsl2> git config --list
                user.name=CoderSuHang
                user.email=1048227620@qq.com
                credential.helper=store
                
                suhang@Y9000P /usr> cd /mnt/e/Software/LinuxOS/wsl2
                suhang@Y9000P /m/e/S/L/wsl2> ssh-keygen -t rsa -C "1048227620@qq.com"
                Generating public/private rsa key pair.
                Enter file in which to save the key (/home/suhang/.ssh/id_rsa):
                Enter passphrase (empty for no passphrase):
                Enter same passphrase again:
                Your identification has been saved in /home/suhang/.ssh/id_rsa
                Your public key has been saved in /home/suhang/.ssh/id_rsa.pub
                The key fingerprint is:
                SHA256:o0XnKXjUJprYQxwXWtqph9LoF9G2A9o+KNY3bo5geqM 1048227620@qq.com
                The key's randomart image is:
                +---[RSA 3072]----+
                |      . +.       |
                |     . O o       |
                |      B O +      |
                |     O % * .     |
                |    = % S o      |
                |   o + B +       |
                |  = + B          |
                | +oo =.o         |
                |Eo ..oo          |
                +----[SHA256]-----+
                suhang@Y9000P /m/e/S/L/wsl2> /home/suhang/.ssh
                suhang@Y9000P ~/.ssh> ll
                total 12K
                -rw------- 1 suhang suhang 2.6K Apr  4 22:49 id_rsa
                -rw-r--r-- 1 suhang suhang  571 Apr  4 22:49 id_rsa.pub
                -rw-r--r-- 1 suhang suhang  142 Apr  4 22:34 known_hosts
                suhang@Y9000P ~/.ssh> vim id_rsa.pub
                ```

      * 根据本机安装的情况，新建Makefile.config文件：

        * ```python
          # Please change the cuda version if needed
          # In default, cuDNN library is located in /usr/local/cuda/lib64
          CXX                         :=  g++
          CUDA_VER                    :=  11.7
          
          # Please modify the opencv and tensorrt install directory
          OPENCV_INSTALL_DIR          :=  /usr/local/include/opencv4
          TENSORRT_INSTALL_DIR        :=  /mnt/e/Software/LinuxOS/wsl2/packages/TensorRT-8.5.1.7
          ```

        * 位置放在了工程目录下：

          * E:\Software\LinuxOS\wsl2\tensorrt_starter\config\Makefile.config

  * 运行成功：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/88a5a24c-b1e2-4d61-9895-003a12f3a91b)


##### （2）配置vsCode

* 打开【main.cpp】文件。在键盘中输入【ctrl+shift+p】调出搜索栏，搜索**configurations(JSON)**：

  * ![image-20240405112145898](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405112145898.png)

* 在【c_cpp_properties.json】文件中加入下面两条指令：

  * ```python
    "configurationProvider": "ms-vscode.makefile-tools",
     "compileCommands": "${workspaceFolder}/compile_commands.json"
    ```

  * ![image-20240405112644829](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405112644829.png)

* 配置language mode

  *  在键盘中输入【ctrl+shift+p】调出搜索栏，搜索**language mode**：

    * 确定C++ --> cpp，CUDA C++ --> cuda-cpp

      * ![image-20240405113254402](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405113254402.png)

    * 如果不是，需要进如vscode language identifier网站查看：

      * [Visual Studio Code language identifiers](https://code.visualstudio.com/docs/languages/identifiers)
      * ![image-20240405113621383](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405113621383.png)
      * ![image-20240405113606627](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405113606627.png)

    * 复制第一条指令，进入刚生成的【.vscode】文件目录下创建【settings.json】文件：

      * ```python
        cd .vscode/
        touch settings.json
        ```

    * 设置cu结尾的文件用cuda-cpp语法：

      * ```python
        {
            "files.associations": {
                "*.cu": "cuda-cpp"
            }
        }
        ```

      * ![image-20240405114129893](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405114129893.png)

* 这样就可以在程序中进行跳转：

  * ![image-20240405114342275](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405114342275.png)

#### 1.5.2 安装必要的插件

##### （1）SSH

如果要远程Ubuntu主机，则需要用SSH插件，我这里是在主机安装的wsl2子系统，所以不需要远程。

![image-20240404201957347](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240404201957347.png)

##### （2）WSL

![image-20240404202932265](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240404202932265.png)

您可以通过打开 WSL 终端、导航到您选择的文件夹并键入【code .】来启动连接到 WSL 的 VS Code 新实例

[如何使用Windows的VScode编辑WSL系统内的文件，Windows与WSL混合交互。（直接使用版）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1S34y1P7KC/?vd_source=0d02ed2f63507c727ce90624d9bd5e6a)

* 在目标目录下也可以code对应文件夹：
  * ![image-20240404204532958](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240404204532958.png)

##### （3）C/C++

![image-20240405170620229](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405170620229.png)

##### （4）C/C++ Extension Pack

![image-20240405170635873](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405170635873.png)

##### （5）vscode-cudacpp

![image-20240405170801458](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405170801458.png)



##### 1.5.3 设置c_cpp_properties.json

##### （1）同1.5.1

* 打开【main.cpp】文件。在键盘中输入【ctrl+shift+p】调出搜索栏，搜索**configurations(JSON)**：

  * ![image-20240405112145898](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405112145898.png)

* 在【c_cpp_properties.json】文件中加入下面两条指令：

  * ```python
    "configurationProvider": "ms-vscode.makefile-tools",
     "compileCommands": "${workspaceFolder}/compile_commands.json"
    ```

  * ![image-20240405112644829](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405112644829.png)

* 配置language mode

  *  在键盘中输入【ctrl+shift+p】调出搜索栏，搜索**language mode**：

    * 确定C++ --> cpp，CUDA C++ --> cuda-cpp

      * ![image-20240405113254402](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405113254402.png)

    * 如果不是，需要进如vscode language identifier网站查看：

      * [Visual Studio Code language identifiers](https://code.visualstudio.com/docs/languages/identifiers)
      * ![image-20240405113621383](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405113621383.png)
      * ![image-20240405113606627](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405113606627.png)

    * 复制第一条指令，进入刚生成的【.vscode】文件目录下创建【settings.json】文件：

      * ```python
        cd .vscode/
        touch settings.json
        ```

    * 设置cu结尾的文件用cuda-cpp语法：

      * ```python
        {
            "files.associations": {
                "*.cu": "cuda-cpp"
            }
        }
        ```

      * ![image-20240405114129893](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405114129893.png)

* 这样就可以在程序中进行跳转：

  * ![image-20240405114342275](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405114342275.png)

#### 1.5.4 设置tasks.json

在我们想要使用GDB调试的时候，是希望在用make执行时候，能够把我们改过的东西同步上去的。

##### （1）创建tasks.json

* 在键盘中输入【ctrl+shift+p】调出搜索栏，搜索**configure task**：
  * ![image-20240405115212805](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405115212805.png)
  * 点击【使用模板创建json文件】，选择【others】：
    * ![image-20240405115339940](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405115339940.png)
    * ![image-20240405115415384](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405115415384.png)

##### （2）编辑tasks.json

* 在【tasks.json】文件中修改指令：

  * ```python
    {
        // See https://go.microsoft.com/fwlink/?LinkId=733558
        // for the documentation about the tasks.json format
        "version": "2.0.0",
        "tasks": [
            {
                "label": "make",
                "type": "shell",
                "command": "make -j16"
            }
        ]
    }
    ```

  * ![image-20240405115816978](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405115816978.png)

#### 1.5.5 设置launch.json

##### （1）设置debug

* 在键盘中输入【ctrl+shift+p】调出搜索栏，搜索**debug: Add Configuration**：
  * ![image-20240405162244501](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405162244501.png)
* 选择【CUDA C++(CUDA-GDB)】选项
  * ![image-20240405162034519](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405162034519.png)
  * 这时候没有【CUDA C++(CUDA-GDB)】选项，原因是没有安装CUDA包，安装下面这个包即可：
    * ![image-20240405163853389](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405163853389.png)
  * 选择即可创建【launch.json】文件
    * ![image-20240405163839009](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405163839009.png)
    * ![image-20240405164050042](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405164050042.png)

##### （2）修改launch.json

* 在program中加入**可执行文件**路径：

  * ```python
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "CUDA C++: Launch",
                "type": "cuda-gdb",
                "request": "launch",
                "program": "${workspaceFolder}/trt-cuda"
            },
            {
                "name": "CUDA C++: Attach",
                "type": "cuda-gdb",
                "request": "attach"
            }
        ]
    }
    ```

  * ![image-20240405164634907](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405164634907.png)

#### 1.5.6 设置setting.json

##### （1）同1.5.1

* 配置language mode

  *  在键盘中输入【ctrl+shift+p】调出搜索栏，搜索**language mode**：

    * 确定C++ --> cpp，CUDA C++ --> cuda-cpp

      * ![image-20240405113254402](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405113254402.png)

    * 如果不是，需要进如vscode language identifier网站查看：

      * [Visual Studio Code language identifiers](https://code.visualstudio.com/docs/languages/identifiers)
      * ![image-20240405113621383](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405113621383.png)
      * ![image-20240405113606627](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405113606627.png)

    * 复制第一条指令，进入刚生成的【.vscode】文件目录下创建【settings.json】文件：

      * ```python
        cd .vscode/
        touch settings.json
        ```

    * 设置cu结尾的文件用cuda-cpp语法：

      * ```python
        {
            "files.associations": {
                "*.cu": "cuda-cpp"
            }
        }
        ```

      * ![image-20240405114129893](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405114129893.png)

* 这样就可以在程序中进行跳转：

  * ![image-20240405114342275](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405114342275.png)

#### 1.5.7 DEBUG

* 用GDB调试main.cpp文件里的断点时，出现以下错误：

  * ```
    No source file named /mnt/e/Software/LinuxOS/wsl2/tensorrt_starter/chapter2-cuda-programming/2.3-matmul-basic/src/main.cpp.
    ```

  * ![image-20240405220749662](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405220749662.png)

  * 问题原因不太清楚，不过这个问题让我知道可执行文件是trt-cuda不是main.cpp

  * 解决办法：

    * 在终端make clean，然后用make DEBUG=1之后可以调试了！
    * ![image-20240405220932173](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405220932173.png)
    * ![image-20240405220946774](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240405220946774.png)










### 1.x 环境配置

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

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e50f64f7-5d3f-452c-94e4-780f1edf8089)


#### 1.3.3 SSH连接

操作系统这里卡了我很久，现在整明白了，需要Windows主机 + Ubuntu服务器，用 SSH 连接。这里参考下面的教程进行配置：

* [VsCode通过SSH连接Ubuntu虚拟机_vscode ssh ubuntu-CSDN博客](https://blog.csdn.net/qq_51607131/article/details/131309223?ops_request_misc=%7B%22request%5Fid%22%3A%22171163374416800222852293%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=171163374416800222852293&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-131309223-null-null.142^v100^pc_search_result_base5&utm_term=vscode用ssh连接ubuntu&spm=1018.2226.3001.4187)

详细配置看视频：

* 【vscode远程连接ubuntu】 https://www.bilibili.com/video/BV1MN411T71b/?share_source=copy_web&vd_source=a0dbe312acd17ef7f1fb082726d496a7

设置免密远程可参考下面视频：

* 【VS Code通过SSH连接Ubuntu进行远程开发，让老电脑起死回生，在局域网中做代码服务器】 https://www.bilibili.com/video/BV15D4y177Ko/?share_source=copy_web&vd_source=a0dbe312acd17ef7f1fb082726d496a7

#### 1.3.4 WinSCP

##### （1）安装教程

* [实用工具系列-WinSCP安装下载与使用-CSDN博客](https://blog.csdn.net/Passerby_Wang/article/details/124913219?ops_request_misc=%7B%22request%5Fid%22%3A%22171179212916800226546693%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=171179212916800226546693&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-124913219-null-null.142^v100^pc_search_result_base5&utm_term=winscp&spm=1018.2226.3001.4187)

##### （2）连接教程

* [文件传输工具WinSCP下载安装教程_winscp安装教程-CSDN博客](https://blog.csdn.net/qq_26383975/article/details/120220823?ops_request_misc=%7B%22request%5Fid%22%3A%22171179212916800226546693%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=171179212916800226546693&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-120220823-null-null.142^v100^pc_search_result_base5&utm_term=winscp&spm=1018.2226.3001.4187)
* 【WinSCP连接远程服务器传输文件】 https://www.bilibili.com/video/BV1v24y1t7RS/?share_source=copy_web&vd_source=a0dbe312acd17ef7f1fb082726d496a7
* 【WinSCP轻松实现文件由Windows传输到Linux】 https://www.bilibili.com/video/BV1tb4y1h71b/?share_source=copy_web&vd_source=a0dbe312acd17ef7f1fb082726d496a7

#### 1.3.5 自动环境配置

##### （1）虚拟环境中的python

* 【首先要在虚拟机环境中安装好Python，这里安装了Python3.8】这里进了很多坑，终于重新安装了一个Ubuntu18.04的虚拟机，并且通过安装pip3实现了trtpy的自动环境安装，来之不易，先这样往后做叭！
  * [在 Ubuntu 上安装 pip的方法_ubuntu pip-CSDN博客](https://blog.csdn.net/baidu_41617231/article/details/135296235)
* [ubuntu 安装python3.8 - CSDN文库](https://wenku.csdn.net/answer/c2c5f58574df7e8844fe0d898eaecabd)
* [Ubuntu 安装Python3.8_ubuntu python3.8-CSDN博客](https://blog.csdn.net/xiaowang_lj/article/details/135679468)

##### （2）安装指令

* ```python
  pip3 install trtpy -U
  ```

##### （3）配置快捷指令

* 这里我安装的是python3，所以不能使用python -m trtpy，应改成python3 -m trtpy使用。
* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ae2416c3-8469-485e-8d01-3120904a8f5b)


* ```python
  echo alias tetpy=\"python3 -m trtpy\" >> ~
  ```

* 
