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

  * ![image-20240321114943113](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240321114943113.png)

* results.panda()方法：

  * ![image-20240321115748138](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240321115748138.png)

* results.crop()方法：

  * ![image-20240321122055441](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240321122055441.png)

* results.print()方法：

  * ![image-20240321122040208](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240321122040208.png)

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

    * ![image-20240321122928345](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240321122928345.png)

    * 原因：在模型检测是调用了判断模型类型的函数：

      * ![image-20240321123149383](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240321123149383.png)

      * ![image-20240321123513083](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240321123513083.png)

      * ![image-20240321123628356](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240321123628356.png)

        * 如果没有names就要从yaml文件中寻找，否则便从0-999生成一系列的class。

        * 所以需要设定一下模型需要用到什么类别以及对应的序号：

          * ```pytho
            model.names = {0: "person", 27: "tie"}
            ```

          * ![image-20240321155542579](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240321155542579.png)
