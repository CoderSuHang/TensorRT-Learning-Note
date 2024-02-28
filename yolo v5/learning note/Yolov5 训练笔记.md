# Yolov5 训练笔记

## 一、配置环境

这里环境大致如下：

* CUDA Toolkit 11.8
* cuDNN v8.9.7 for CUDA 11.x
* Anaconda3 -2021.05
* Pycharm 2022.1.3

有变化的是安装了up主推荐的pytorch v1.8.2 cuda11.1版本：

* 【【手把手带你实战YOLOv5-入门篇】YOLOv5 环境安装（重置版）】 【精准空降到 12:14】 https://www.bilibili.com/video/BV1bg4y1R7cs/?share_source=copy_web&vd_source=a0dbe312acd17ef7f1fb082726d496a7&t=734
* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b6904c44-053a-4e8c-a5a6-61f0cd93d0d6)

修改yolov5源代码：

* [Release v7.0 - YOLOv5 SOTA Realtime Instance Segmentation · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/releases/tag/v7.0)
* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7fc91398-def1-405c-93a2-707b5326e5ef)

安装yolov5：

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7f325871-7aa7-459d-a02b-b64aca4a7f34)

运行yolov5:

* 输入指令：
  * python detect.py
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5ef4ef96-7c7e-473b-aad4-3e08df6a8ffd)
* 输出结果：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a840d15b-78ef-4514-a2cf-ee97d9ddce9b)
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7c3c4896-075c-462a-8b63-73717dd2164c)

## 二、模型检测

参考资料：

​	[【手把手带你实战YOLOv5-入门篇】YOLOv5 模型检测_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1B8411c7ZN/?spm_id_from=333.788&vd_source=0d02ed2f63507c727ce90624d9bd5e6a)

### 2.1 关键参数

* weights：训练好的模型文件

  * ![image-20240228122333216](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240228122333216.png)

  * 指令：

    * ```python
      (yolov5) E:\AI\Package\yolov5-7.0>python detect.py --weights yolov5s.pt
      ```

      * ![image-20240228122754959](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240228122754959.png)

    * ```python
      (yolov5) E:\AI\Package\yolov5-7.0>python detect.py --weights yolov5x.pt
      ```

      * ![image-20240228123102594](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240228123102594.png)

* source：检测的目标，可以是单张图片、文件夹、屏幕或者是摄像头：

  * ```python
    python path/to/detect.py --weights yolov5s.pt --source 0              # webcam # 直播软件/电脑摄像头
                                                           img.jpg        # image
                                                           vid.mp4        # video
            										   screen         # screenshot
                                                           path/          # directory
                    								    list.txt	   # list of images
                        							   	list.streams   # list of streams
                                                           'path/*.jpg'   # glob
                                                           'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                           'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
    ```

  * ```python
    (yolov5) E:\AI\Package\yolov5-7.0>python detect.py --weights yolov5s.pt --source data/images/bus.jpg
    ```

    * ![image-20240228153003733](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240228153003733.png)

* conf-thres：置信度阈值，越低框越多，越高框越少

  * ![image-20240228153605515](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240228153605515.png)

* iou-thres：IOU阈值，越低框越少，越高框越多

  * ![image-20240228154025110](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240228154025110.png)



2.2 使用torch

* 安装方法按照视频里面教的来。

* 代码示例：

  * ```python
    import torch
    
    # Model 加载本地的模型
    model = torch.hub.load("./", "yolov5s", source="local")
    
    # Images 指定检测图片
    img = "./data/images/zidane.jpg"
    
    # Inference
    results = model(img)
    
    # Results
    results.show()
    ```

  * ![image-20240228155155790](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240228155156921.png)

