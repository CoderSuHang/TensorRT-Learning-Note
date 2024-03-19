这里已经安装完成了CUDA 11.8、cuDNN 11.X:

* CUDA Toolkit 11.8
* cuDNN v8.9.7 for CUDA 11.x

现在安装TensorRT和zlibwapo.dll：

### 7.1 TensorRT安装

#### 7.1.1 安装包下载

* 网址：[NVIDIA TensorRT Download | NVIDIA Developer](https://developer.nvidia.com/tensorrt-download)
* 选择TensorRT 8.5 GA：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d9465634-1c41-4c13-8a6e-be41fde5bcef)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/00758abf-6d92-4bac-a08b-bb87f56fd47a)


#### 7.1.2 根据自己的python版本安装

* 查看python版本3.8.18：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c08fe02e-aa9e-4112-954a-bccf6e7c2e96)

* 安装对应版本的.whl文件：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f5053694-b28c-42e0-8173-c422c2e0ad71)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c74ddb0e-bd84-436d-a4cf-e686eccd65bf)

* 将【lib】文件夹下的动态链接库拷贝至CUDA安装位置的【bin】文件夹中：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b3aa1569-4a8e-462f-8aa9-8595beb777ae)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/694af55f-44d0-4cb8-a825-4bb160316112)


### 7.2 zlibwapi.dll安装

（安装cu116版本的pytorch可以避免此问题）

#### 7.2.1 安装包下载

* 网址：[ZLIB DLL Home Page (winimage.com)](http://www.winimage.com/zLibDll/)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/effbfbb4-8d52-469e-80ba-5490dc618969)


#### 7.2.2 解压将dll文件复制道CUDA的【bin】文件夹内：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/cd8c1d88-2820-489b-893c-6b606b4ee0c9)

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/dd042ebe-aa9b-4d77-bb71-4c97dc0a9b6f)


### 7.3 onnx安装与配置

onnx类似于适配器，模型需要用原始状态借助oonx这个桥梁转成目标状态。在vsCode中进行安装即可：

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e85248c0-5605-473c-ae6d-d6704de458c1)


### 7.4 模型导出

* 运行指令：

  * ```python
    python export.py --weights yolov5s.pt --include engine --device 0
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f97e62b4-57bd-4c74-9817-38116c0df5b0)


    * 报错：

      * ```python
        TensorRT: export failure  0.0s: [WinError 127]
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a8733180-f6c9-4860-b74a-5a15de4886fd)


    * 解决方案：

      * [Why my model export into TensorRT engine but response with an OS Error? · Issue #10706 · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/issues/10706)
      * 使用annconda虚拟环境中的cublas64_11.dll文件替换掉cuda环境中的cublas64_11.dll文件。
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d757e308-2893-4d05-aea4-f3e8a892bd9f)

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/adff3140-9202-46c7-9114-a90edbdf4779)


* 运行成功：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0d433480-a834-46f4-ba71-c177690497ea)


* 在【detect.py】中运行TensorRT的模型：

  * python detect.py --weights yolov5s.engine
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/95381639-c92c-4dcb-a548-246e9ffa704c)

