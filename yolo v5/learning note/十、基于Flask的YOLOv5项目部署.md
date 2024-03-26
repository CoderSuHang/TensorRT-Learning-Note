### 10.1 基于文件的预测方式

```python
utils/flask_rest_api 中为官方示例
```

![image-20240326193418914](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240326193418914.png)

* 加载本地模型：
  * ![image-20240326193734705](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240326193734705.png)
* 实现步骤：
  * （1）将flask中的两个py文件粘贴到yolov5的根目录下：
    * ![image-20240326194042380](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240326194042380.png)
  * （2）修改【example_request.py】文件：
    * 修改图片文件位置信息：
      * ![image-20240326200453343](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240326200453343.png)
  * （3）修改【restapi.py】文件：
    * 模型加载目录修改为当前目录下：
      * ![image-20240326194917073](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240326194917073.png)
  * （4）启动【restapi.py】：
    * 报错：
      * ![image-20240326195104280](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240326195104280.png)
      * 未安装flask，pip指令安装一下即可：
      * ![image-20240326200111907](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240326200111907.png)
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

