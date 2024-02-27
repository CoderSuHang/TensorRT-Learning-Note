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
  * ![bus](E:\AI\Package\yolov5-7.0\runs\detect\exp2\bus.jpg)
  * ![zidane](E:\AI\Package\yolov5-7.0\runs\detect\exp2\zidane.jpg)

## 二、模型检测
