### 9.1 模型加载

* 文件位置：

  * 【hub_detect.ipynb】

* 代码：

  * ```python
    import torch
    import time
    
    # Model
    model = torch.hub.load("./", "custom", path="yolov5s.pt", source="local")
    ```

### 9.2 模型预测

* 代码：

  * ```python
    # imgs = [
    #         'data/images/zidane.jpg',  # filename
    #         Path('data/images/zidane.jpg'),  # Path
    #         'https://ultralytics.com/images/zidane.jpg',  # URI
    #         cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
    #         Image.open('data/images/bus.jpg'),  # PIL
    #         np.zeros((320, 640, 3))]  # numpy
    # Images
    img = "./data/images/zidane.jpg"
    
    results = model(img)
    ```

### 9.3 预测结果

* 代码：

  * ```python
    results.show()		# 图片可视化
    results.render()	# 图片结果
    results.pandas()	# 数据结果
    results.crops()		# 裁切结果
    results.print()		# 文本结果
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b8fb3066-75ff-4cd5-9e2a-611661d3855e)


* results.panda()方法：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c27979ac-1090-4170-a30e-e79f6dd06a4c)


* results.crop()方法：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a5fde659-2136-48b5-a37e-aa202d9e2c18)


* results.print()方法：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/84ac15f0-7af0-4481-8e2e-9f598da9e731)


### 9.4 使用TensorRT模型预测

* 在模型加载时修改成engine文件名即可：

  * ```python
    import torch
    import time
    
    # Model
    model = torch.hub.load("./", "custom", path="yolov5s-fp32.engine", source="local")
    ```

* 运行报错：

  * 类别名称消失：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0c08167b-cd2e-4657-a0ee-ec7db1063dd8)


    * 原因：在模型检测是调用了判断模型类型的函数：

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e9b79862-0062-4155-9a2d-1fdadfb2df54)


      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0724ac1c-0359-44a7-a4cf-e01c36368978)


      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f11ec980-cc11-48b6-ad0b-1bc8cee429f3)

        * 如果没有names就要从yaml文件中寻找，否则便从0-999生成一系列的class。

        * 所以需要设定一下模型需要用到什么类别以及对应的序号：

          * ```pytho
            model.names = {0: "person", 27: "tie"}
            ```

          * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/85f84a9c-6faa-4837-a209-59455d266b6f)

