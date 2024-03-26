### 10.1 基于文件的预测方式

```python
utils/flask_rest_api 中为官方示例
```

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a7fb8b36-6ac5-4c46-89b3-56291bb20ff1)


* 加载本地模型：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c1a1e7db-66ad-4407-92e4-d6c467fc1905)

* 实现步骤：
  * （1）将flask中的两个py文件粘贴到yolov5的根目录下：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d4af8c3b-d689-43c1-8cc7-34ea8331129d)

  * （2）修改【example_request.py】文件：
    * 修改图片文件位置信息：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/90783e9e-8171-4cc0-b491-c21f19194744)

  * （3）修改【restapi.py】文件：
    * 模型加载目录修改为当前目录下：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/183c3c45-b987-4647-b718-8c48ab61bca5)

  * （4）启动【restapi.py】：
    * 报错：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0123d86e-816c-4d20-bd45-84f09da22d61)

      * 未安装flask，pip指令安装一下即可：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2b90c9f1-ffb5-49b5-8ea6-b2c4a7f7a790)

    * ![image-20240326200611384](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240326200611384.png)
  * （5）接着运行【example_request.py】：
    * ![image-20240326200555403](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240326200555403.png)
    * 即可看到预测结果。

### 10.2 基于图像的预测方式

```python
cv2.imencode.cv2.imdecode 搭配使用
cv.2imencode(".jpg", img)[1].tobytes()
cv.2imdecode(np.frombuffer(xxx, dtype=np.uint8), cv2.IMREAD_COLOR)
```

