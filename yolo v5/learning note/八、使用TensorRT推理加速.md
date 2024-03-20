### 8.1 加速对比

#### 8.1.1 原始detect

* 指令：

  * ```python
    python detect.py --weights yolov5s.pt
    ```

* 速度：

  * ![image-20240320094334056](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240320094334056.png)

#### 8.1.2 加速后detect

* 指令：

  * ```python
    python detect.py --weights yolov5s.engine
    ```

* 速度：

  * ![image-20240320094736267](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240320094736267.png)

* 效果：

  * ![image-20240320100814037](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240320100814037.png)

  * 速度并没有提升，反而有所缓慢，原因是图像输入维度不同，Ptorch推理是386x640，而TensoRT是640x640，所以推理速度会过慢。

  * 而TensorRT在模型导出过程中，就已经把输入维度固定死了，即使在运行模型检测的时候强制修改输入维度，也无法正常运行TensorRT的模型去检测目标维度图像。因此，在导出模型的过程中就需要确定号目标检测的维度大小：

    * ```python
      python export.py --weights yolov5s.pt --include engine --device 0 --img 384 640
      ```

    * ![image-20240320102755599](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240320102755599.png)

    * ![image-20240320102906844](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240320102906844.png)

  * 这时便可以指定大小的输入维度进行运行：

    * ```python
      python detect.py --weights yolov5s.engine --imgsz 384 640
      ```

    * ![image-20240320103146110](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240320103146110.png)

    * ![image-20240320103209557](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240320103209557.png)

  * 由此可见使用TensorRT推理加速后的模型速度会有更大提升：

    * ```python
      # PT
      video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 9.0ms
      video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 10.0ms
      video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 9.0ms
      video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 10.0ms
      video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 9.0ms
      video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 8.0ms
      Speed: 0.3ms pre-process, 8.7ms inference, 1.1ms NMS per image at shape (1, 3, 640, 640)
      Results saved to runs\detect\exp11
      
      
      # TR
      video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 1 tie, 5 apples, 2 oranges, 3.0ms
      video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 1 tie, 5 apples, 1 orange, 6.0ms
      video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 1 tie, 5 apples, 2 oranges, 7.1ms
      video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 5 apples, 1 orange, 5.0ms
      video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 5 apples, 1 orange, 8.5ms
      video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 5 apples, 1 orange, 3.2ms
      Speed: 0.5ms pre-process, 5.0ms inference, 1.0ms NMS per image at shape (1, 3, 640, 640)
      Results saved to runs\detect\exp12
      
      # TRnew
      video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 6.2ms
      video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 3.6ms
      video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 4.7ms
      video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 4.0ms
      video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 4.8ms
      video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 3.4ms
      Speed: 0.4ms pre-process, 4.0ms inference, 0.9ms NMS per image at shape (1, 3, 384, 640)
      Results saved to runs\detect\exp14
      ```

    * inference推理时间落在了4ms。

### 8.2 进一步加速推理

（1）【export.py】文件中的half参数：

* 表示是否用半精度进行推理

* 推理：

  * ```python
    python export.py --weights yolov5s.pt --include engine --device 0 --img 384 640 --half
    ```

* 检测：

  * ```python
    python detect.py --weights yolov5s-fp16.engine --imgsz 384 640 --half
    ```

  * ![image-20240320104946439](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240320104946439.png)

  * ```python
    # PT
    video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 9.0ms
    video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 10.0ms
    video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 9.0ms
    video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 10.0ms
    video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 9.0ms
    video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 8.0ms
    Speed: 0.3ms pre-process, 8.7ms inference, 1.1ms NMS per image at shape (1, 3, 640, 640)
    Results saved to runs\detect\exp11
    
    
    # TR
    video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 1 tie, 5 apples, 2 oranges, 3.0ms
    video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 1 tie, 5 apples, 1 orange, 6.0ms
    video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 1 tie, 5 apples, 2 oranges, 7.1ms
    video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 5 apples, 1 orange, 5.0ms
    video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 5 apples, 1 orange, 8.5ms
    video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 5 apples, 1 orange, 3.2ms
    Speed: 0.5ms pre-process, 5.0ms inference, 1.0ms NMS per image at shape (1, 3, 640, 640)
    Results saved to runs\detect\exp12
    
    # TRnew
    video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 6.2ms
    video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 3.6ms
    video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 4.7ms
    video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 4.0ms
    video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 4.8ms
    video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 3.4ms
    Speed: 0.4ms pre-process, 4.0ms inference, 0.9ms NMS per image at shape (1, 3, 384, 640)
    Results saved to runs\detect\exp14
    
    # TRhalf
    video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 8.7ms
    video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 7.0ms
    video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 8.4ms
    video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 5.0ms
    video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 7.1ms
    video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 5.0ms
    Speed: 0.6ms pre-process, 3.8ms inference, 1.3ms NMS per image at shape (1, 3, 384, 640)
    Results saved to runs\detect\exp18
    ```

（2）如果不要求精度，还可以按照图像比例，进一步缩小imgsz维度：

推理：

* ```python
  python export.py --weights yolov5s.pt --include engine --device 0 --img 192 320 --half
  ```

检测：

* ```python
  python detect.py --weights yolov5s-halfsize.engine --imgsz 192 320 --half
  ```

* ![image-20240320111318123](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240320111318123.png)

* ```python
  # PT
  video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 9.0ms
  video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 10.0ms
  video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 9.0ms
  video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 10.0ms
  video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 9.0ms
  video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 8.0ms
  Speed: 0.3ms pre-process, 8.7ms inference, 1.1ms NMS per image at shape (1, 3, 640, 640)
  Results saved to runs\detect\exp11
  
  
  # TR
  video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 1 tie, 5 apples, 2 oranges, 3.0ms
  video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 1 tie, 5 apples, 1 orange, 6.0ms
  video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 1 tie, 5 apples, 2 oranges, 7.1ms
  video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 5 apples, 1 orange, 5.0ms
  video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 5 apples, 1 orange, 8.5ms
  video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 640x640 1 person, 5 apples, 1 orange, 3.2ms
  Speed: 0.5ms pre-process, 5.0ms inference, 1.0ms NMS per image at shape (1, 3, 640, 640)
  Results saved to runs\detect\exp12
  
  # TRnew
  video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 6.2ms
  video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 3.6ms
  video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 4.7ms
  video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 4.0ms
  video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 4.8ms
  video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 3.4ms
  Speed: 0.4ms pre-process, 4.0ms inference, 0.9ms NMS per image at shape (1, 3, 384, 640)
  Results saved to runs\detect\exp14
  
  # TRhalf
  video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 8.7ms
  video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 7.0ms
  video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 8.4ms
  video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 5.0ms
  video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 7.1ms
  video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 384x640 1 person, 1 tie, 5 apples, 1 orange, 5.0ms
  Speed: 0.6ms pre-process, 3.8ms inference, 1.3ms NMS per image at shape (1, 3, 384, 640)
  Results saved to runs\detect\exp18
  
  # TRhalfsize
  video 1/1 (1088/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 192x320 1 person, 1 orange, 2.5ms
  video 1/1 (1089/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 192x320 1 person, 1 orange, 1.6ms
  video 1/1 (1090/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 192x320 1 person, 1 orange, 2.1ms
  video 1/1 (1091/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 192x320 1 person, 1 orange, 1.3ms
  video 1/1 (1092/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 192x320 1 person, 1 orange, 2.9ms
  video 1/1 (1093/1094) E:\AI\Package\yolov5-7.0\datasets\gesture.mp4: 192x320 1 person, 1 orange, 2.0ms
  Speed: 0.3ms pre-process, 2.0ms inference, 1.1ms NMS per image at shape (1, 3, 192, 320)
  Results saved to runs\detect\exp20
  ```

### 8.3 推理加速需要注意的地方

* 使用TensorRT推理时，在【export.py】导出模型这一步，一定要注意它的输入维度要跟我们最终要预测的图像维度对齐，只有这样才能保证加速有效。

