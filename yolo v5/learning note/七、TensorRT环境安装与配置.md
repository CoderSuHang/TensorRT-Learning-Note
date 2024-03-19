这里已经安装完成了CUDA 11.8、cuDNN 11.X:

* CUDA Toolkit 11.8
* cuDNN v8.9.7 for CUDA 11.x

现在安装TensorRT和zlibwapo.dll：

### 7.1 TensorRT安装

#### 7.1.1 安装包下载

* 网址：[NVIDIA TensorRT Download | NVIDIA Developer](https://developer.nvidia.com/tensorrt-download)
* 选择TensorRT 8.5 GA：
  * ![image-20240319105814442](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319105814442.png)
  * ![image-20240319110323383](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319110323383.png)

#### 7.1.2 根据自己的python版本安装

* 查看python版本3.8.18：
  * ![image-20240319112851216](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319112851216.png)
* 安装对应版本的.whl文件：
  * ![image-20240319112931635](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319112931635.png)
  * ![image-20240319113643265](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319113643265.png)
* 将【lib】文件夹下的动态链接库拷贝至CUDA安装位置的【bin】文件夹中：
  * ![image-20240319113901641](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319113901641.png)
  * ![image-20240319114017174](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319114017174.png)

### 7.2 zlibwapi.dll安装

（安装cu116版本的pytorch可以避免此问题）

#### 7.2.1 安装包下载

* 网址：[ZLIB DLL Home Page (winimage.com)](http://www.winimage.com/zLibDll/)
  * ![image-20240319114530210](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319114530210.png)

#### 7.2.2 解压将dll文件复制道CUDA的【bin】文件夹内：

* ![image-20240319114740558](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319114740558.png)
* ![image-20240319114946695](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319114946695.png)

### 7.3 onnx安装与配置

onnx类似于适配器，模型需要用原始状态借助oonx这个桥梁转成目标状态。在vsCode中进行安装即可：

![image-20240319115837707](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319115837707.png)

### 7.4 模型导出

* 运行指令：

  * ```python
    python export.py --weights yolov5s.pt --include engine --device 0
    ```

  * ![image-20240319165954529](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319165954529.png)

    * 报错：

      * ```python
        TensorRT: export failure  0.0s: [WinError 127]
        ```

      * ![image-20240319170036839](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319170036839.png)

    * 解决方案：

      * [Why my model export into TensorRT engine but response with an OS Error? · Issue #10706 · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/issues/10706)
      * 使用annconda虚拟环境中的cublas64_11.dll文件替换掉cuda环境中的cublas64_11.dll文件。
      * ![image-20240319171610287](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319171610287.png)
      * ![image-20240319171758645](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319171758645.png)

* 运行成功：

  * ![image-20240319172338641](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319172338641.png)

* 在【detect.py】中运行TensorRT的模型：

  * python detect.py --weights yolov5s.engine
  * ![image-20240319172819399](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240319172819399.png)
