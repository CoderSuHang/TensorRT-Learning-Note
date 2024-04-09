## 二、CUDA编程入门

### 2.1 CUDA中的线程与线程束

#### 2.1.1 执行第一个CUDA程序

##### （1）环境配置

见1.5

##### （2）make

* 在终端中先make
  * ![image-20240409102422551](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409102422551.png)

##### （3）trt-cuda

* 再运行trt-cuda的可执行文件
  * ![image-20240409102447198](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409102447198.png)

#### 2.1.2 理解CUDA中的Grid和Block

![image-20240409102528215](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409102528215.png)

* 其中Host相当于我们的CPU，Device相当于GPU，我们一般写程序的时候是从CPU开始执行的，Kernel是一个核函数（以线程为单位计算的函数），它会调用Device中的Grid。
* 一个Kernel对应一个Grid，一个Grid对应多个Block，一个Block中又对应多个Thread
* Grid，Block都属于大量Thread的组合

##### （1）Grid和Block的关联

* 一个Grid里面有多个Block，一个Block中的多个Thread有自己的Registers和Local Memory。
* 在同一个Block里的Thread共享同一个Shared Memory；
* 在同一个Grid里的所有Block公用同一个Global Memory、Constant Memory、Texture Memory。
* Thread距离这些Memory的距离决定了他们的访问速度（金字塔结构）。

##### （2）block和thread的遍历(traverse)

###### 1. 一维

* 【一维定义】

  * 例如：把一个8个数据大小的数组分为两个block进行访问，一个block有四个不用的thread的访问地址：

  * ![image-20240409110857665](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409110857665.png)

  * ```c++
    void print_one_dim(){
        int inputSize = 8;	// 数据大小为8
        int blockDim = 4;	// 一个block有4个Thread
        int gridDim = inputSize / blockDim;		//grid
    
        dim3 block(blockDim);
        dim3 grid(gridDim);
    
        /* 这里建议大家吧每一函数都试一遍*/
        print_idx_kernel<<<grid, block>>>();
        // print_dim_kernel<<<grid, block>>>();
        // print_thread_idx_per_block_kernel<<<grid, block>>>();
        // print_thread_idx_per_grid_kernel<<<grid, block>>>();
    
        cudaDeviceSynchronize();
    }
    ```

  * 输出：

    * ```c++
      // 核函数之前一定是加（__global__）前缀，表示它是一个核函数
      __global__ void print_idx_kernel(){
          printf("block idx: (%3d, %3d, %3d), thread idx: (%3d, %3d, %3d)\n",
               blockIdx.z, blockIdx.y, blockIdx.x,
               threadIdx.z, threadIdx.y, threadIdx.x);
      }
      ```

      * ![image-20240409112334193](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409112334193.png)

    * ```c++
      //打印各个维度上grid和block的大小
      __global__ void print_dim_kernel(){
          printf("grid dimension: (%3d, %3d, %3d), block dimension: (%3d, %3d, %3d)\n",
               gridDim.z, gridDim.y, gridDim.x,
               blockDim.z, blockDim.y, blockDim.x);
      }
      ```

      * ![image-20240409113355252](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409113355252.png)

* 【一维索引】

  * 在Block空间下寻找Thread ID

    * ```c++
      __global__ void print_thread_idx_per_block_kernel(){
          int index = threadIdx.z * blockDim.x * blockDim.y + \
                    threadIdx.y * blockDim.x + \
                    threadIdx.x;
      
          printf("block idx: (%3d, %3d, %3d), thread idx: %3d\n",
               blockIdx.z, blockIdx.y, blockIdx.x,
               index);
      }
      ```

      * ![image-20240409113858214](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409113858214.png)

    * ![image-20240409114140801](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409114140801.png)

  * 在Grid空间下寻找Thread ID

    * 先找Block ID（和找Thread ID方法一致），再找Thread ID

    * ```C++
      __global__ void print_thread_idx_per_grid_kernel(){
          int bSize  = blockDim.z * blockDim.y * blockDim.x;
      
          int bIndex = blockIdx.z * gridDim.x * gridDim.y + \
                     blockIdx.y * gridDim.x + \
                     blockIdx.x;
      
          int tIndex = threadIdx.z * blockDim.x * blockDim.y + \
                     threadIdx.y * blockDim.x + \
                     threadIdx.x;
      
          int index  = bIndex * bSize + tIndex;
      
          printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n", 
               bIndex, tIndex, index);
      }
      ```

      * ![image-20240409115237992](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409115237992.png)

    * ![image-20240409115106157](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409115106157.png)

###### 2. 二维

* 二维定义：

  * 例如：

    * ```c++
      void print_two_dim(){
          int inputWidth = 4;		// 数据宽度为4，在二维数组中数据的大小就为4x4=16
      
          int blockDim = 2;		// 一个block里面有2x2=4个thread
          int gridDim = inputWidth / blockDim;	// gird=（4/2）^2=4个block
      
          dim3 block(blockDim, blockDim); // 一个block有（blockDim^2）个thread
          dim3 grid(gridDim, gridDim);	// 一个grid有（gridDim^2）个block
      
          /* 这里建议大家吧每一函数都试一遍*/
          // print_idx_kernel<<<grid, block>>>();
          // print_dim_kernel<<<grid, block>>>();
          // print_thread_idx_per_block_kernel<<<grid, block>>>();
          print_thread_idx_per_grid_kernel<<<grid, block>>>();
      
          cudaDeviceSynchronize();
      }
      ```

  * 输出：

    * ![image-20240409120414407](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409120414407.png)
    * ![image-20240409120526483](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409120526483.png)

* 二维索引：

  * ![image-20240409120620884](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409120620884.png)
  * ![image-20240409120657815](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409120657815.png)

##### （3）cudaDeviceSynchronize

这是一个同步函数，用在打印数据的时候：

* ```c++
  void print_one_dim(){
      int inputSize = 8;
      int blockDim = 4;
      int gridDim = inputSize / blockDim;
  
      dim3 block(blockDim);
      dim3 grid(gridDim);
  
      /* 这里建议大家吧每一函数都试一遍*/
      // print_idx_kernel<<<grid, block>>>();
      // print_dim_kernel<<<grid, block>>>();
      // print_thread_idx_per_block_kernel<<<grid, block>>>();
      print_thread_idx_per_grid_kernel<<<grid, block>>>();
  
      cudaDeviceSynchronize();
      /*
      synchronize是同步的意思，有几种synchronize
  
      cudaDeviceSynchronize: CPU与GPU端完成同步，CPU不执行之后的语句，知道这个语句以前的所有cuda操作结束
      cudaStreamSynchronize: 跟cudaDeviceSynchronize很像，但是这个是针对某一个stream的。只同步指定的stream中的cpu/gpu操作，其他的不管
      cudaThreadSynchronize: 现在已经不被推荐使用的方法
      __syncthreads:         线程块内同步
      */
  }
  ```

* cudaDeviceSynchronize: CPU与GPU端完成同步，CPU不执行之后的语句，知道这个语句以前的所有cuda操作结束。（一般在CPU端执行到一个核函数的时候，不会等核函数执行完的结果，而是立即执行下一个函数）

* 如果不适用这个同步函数，执行代码的结果：

  * ![image-20240409120017459](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240409120017459.png)

（4）print_cord();

* 打印坐标

  * ```c++
    __global__ void print_cord_kernel(){
        int index = threadIdx.z * blockDim.x * blockDim.y + \
                  threadIdx.y * blockDim.x + \
                  threadIdx.x;
    
        int x  = blockIdx.x * blockDim.x + threadIdx.x;
        int y  = blockIdx.y * blockDim.y + threadIdx.y;
    
        printf("block idx: (%3d, %3d, %3d), thread idx: %3d, cord: (%3d, %3d)\n",
             blockIdx.z, blockIdx.y, blockIdx.x,
             index, x, y);
    }
    ```

    * 
