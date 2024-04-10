## 二、CUDA编程入门

### 2.1 CUDA中的线程与线程束

#### 2.1.1 执行第一个CUDA程序

##### （1）环境配置

见1.5

##### （2）make

* 在终端中先make
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7bfac5a7-c68f-44ad-aa0e-f8d06ac212d5)


##### （3）trt-cuda

* 再运行trt-cuda的可执行文件
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/90b928e4-f5e0-47d6-8d75-61570fdc1bff)


#### 2.1.2 理解CUDA中的Grid和Block

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/54cbc931-44eb-4c4a-892a-93005e48be8b)


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

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/50a0c205-ac0a-446f-bb59-d875d3a284bb)


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

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ba2fb1d9-6105-478c-aecb-8f98304ca5b2)


    * ```c++
      //打印各个维度上grid和block的大小
      __global__ void print_dim_kernel(){
          printf("grid dimension: (%3d, %3d, %3d), block dimension: (%3d, %3d, %3d)\n",
               gridDim.z, gridDim.y, gridDim.x,
               blockDim.z, blockDim.y, blockDim.x);
      }
      ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a7f5f12d-eeef-4818-9068-1c6eda3d6713)


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

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bf8512aa-a86b-46ae-8eda-7b04f0adf24c)


    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/eeb4d04a-e6bc-44e0-a2dd-598a44dc0a91)


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

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/795c1555-93b1-4db0-bd7a-cf6f0562039f)


    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c1d89a5a-c0f5-41b6-b542-868f90d948a9)


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

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7baea5e8-8754-4a8f-9b6c-7424afa1cdc6)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7e3328c3-0461-47a5-a71c-5e9b1fe28163)


* 二维索引：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/cf24bc1e-3035-42f4-839f-5989b2cc8b57)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/fae47454-e66b-47ba-a82d-0264dd556592)


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

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f5c247ad-c46c-4778-964c-62650f1bb10f)


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

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b45f989c-36cd-426b-b479-c79fa81cd99f)


  * 输出：

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4884d66f-bc8e-4439-a1f9-0bd4bcc5ce57)

#### 2.1.3 理解.cu和.cpp的相互引用及Makefile

##### （1）操着指令

* 导出操作指令文本命令：

  * ```python
    suhang@Y9000P /m/e/S/L/w/t/c/2/src (main)> g++ --help > g++_compile_options.txt
    suhang@Y9000P /m/e/S/L/w/t/c/2/src (main)> nvcc --help > nvcc__compile_options.txt
    ```

* g++编译操作指令：

  * ```
    Usage: g++ [options] file...
    Options:
      -pass-exit-codes         Exit with highest error code from a phase.
      --help                   Display this information.
      --target-help            Display target specific command line options.
      --help={common|optimizers|params|target|warnings|[^]{joined|separate|undocumented}}[,...].
                               Display specific types of command line options.
      (Use '-v --help' to display command line options of sub-processes).
      --version                Display compiler version information.
      -dumpspecs               Display all of the built in spec strings.
      -dumpversion             Display the version of the compiler.
      -dumpmachine             Display the compiler's target processor.
      -print-search-dirs       Display the directories in the compiler's search path.
      -print-libgcc-file-name  Display the name of the compiler's companion library.
      -print-file-name=<lib>   Display the full path to library <lib>.
      -print-prog-name=<prog>  Display the full path to compiler component <prog>.
      -print-multiarch         Display the target's normalized GNU triplet, used as
                               a component in the library path.
      -print-multi-directory   Display the root directory for versions of libgcc.
      -print-multi-lib         Display the mapping between command line options and
                               multiple library search directories.
      -print-multi-os-directory Display the relative path to OS libraries.
      -print-sysroot           Display the target libraries directory.
      -print-sysroot-headers-suffix Display the sysroot suffix used to find headers.
      -Wa,<options>            Pass comma-separated <options> on to the assembler.
      -Wp,<options>            Pass comma-separated <options> on to the preprocessor.
      -Wl,<options>            Pass comma-separated <options> on to the linker.
      -Xassembler <arg>        Pass <arg> on to the assembler.
      -Xpreprocessor <arg>     Pass <arg> on to the preprocessor.
      -Xlinker <arg>           Pass <arg> on to the linker.
      -save-temps              Do not delete intermediate files.
      -save-temps=<arg>        Do not delete intermediate files.
      -no-canonical-prefixes   Do not canonicalize paths when building relative
                               prefixes to other gcc components.
      -pipe                    Use pipes rather than intermediate files.
      -time                    Time the execution of each subprocess.
      -specs=<file>            Override built-in specs with the contents of <file>.
      -std=<standard>          Assume that the input sources are for <standard>.
      --sysroot=<directory>    Use <directory> as the root directory for headers
                               and libraries.
      -B <directory>           Add <directory> to the compiler's search paths.
      -v                       Display the programs invoked by the compiler.
      -###                     Like -v but options quoted and commands not executed.
      -E                       Preprocess only; do not compile, assemble or link.
      -S                       Compile only; do not assemble or link.
      -c                       Compile and assemble, but do not link.
      -o <file>                Place the output into <file>.
      -pie                     Create a dynamically linked position independent
                               executable.
      -shared                  Create a shared library.
      -x <language>            Specify the language of the following input files.
                               Permissible languages include: c c++ assembler none
                               'none' means revert to the default behavior of
                               guessing the language based on the file's extension.
    
    Options starting with -g, -f, -m, -O, -W, or --param are automatically
     passed on to the various sub-processes invoked by g++.  In order to pass
     other options on to these processes the -W<letter> options must be used.
    
    For bug reporting instructions, please see:
    <file:///usr/share/doc/gcc-11/README.Bugs>.
    
    ```

* nvcc编译操作指令：

  * ```
    Usage  : nvcc [options] <inputfile>
    
    Options for specifying the compilation phase
    ============================================
    More exactly, this option specifies up to which stage the input files must be compiled,
    according to the following compilation trajectories for different input file types:
            .c/.cc/.cpp/.cxx : preprocess, compile, link
            .o               : link
            .i/.ii           : compile, link
            .cu              : preprocess, cuda frontend, PTX assemble,
                               merge with host C code, compile, link
            .gpu             : cicc compile into cubin
            .ptx             : PTX assemble into cubin.
    
    --cuda                                          (-cuda)                         
            Compile all .cu input files to .cu.cpp.ii output.
    
    --cubin                                         (-cubin)                        
            Compile all .cu/.gpu/.ptx input files to device-only .cubin files.  This
            step discards the host code for each .cu input file.
    
    --fatbin                                        (-fatbin)                       
            Compile all .cu/.gpu/.ptx/.cubin input files to device-only .fatbin files.
            This step discards the host code for each .cu input file.
    
    --ptx                                           (-ptx)                          
            Compile all .cu input files to device-only .ptx files.  This step discards
            the host code for each of these input file.
    
    --optix-ir                                      (-optix-ir)                     
            Compile CUDA source to OptiX IR (.optixir) output. The OptiX IR is only intended
            for consumption by OptiX through appropriate APIs. This feature is not supported
            with link-time-optimization (-dlto), the lto_NN -arch target, or with -gencode.
    
    --preprocess                                    (-E)                            
            Preprocess all .c/.cc/.cpp/.cxx/.cu input files.
    
    --generate-dependencies                         (-M)                            
            Generate a dependency file that can be included in a make file for the .c/.cc/.cpp/.cxx/.cu
            input file. If -MF is specified, multiple source files are not supported,
            and the output is written to the specified file, otherwise it is written
            to stdout.
    
    --generate-nonsystem-dependencies               (-MM)                           
            Same as --generate-dependencies but skip header files found in system directories
            (Linux only).
    
    --generate-dependencies-with-compile            (-MD)                           
            Generate a dependency file and compile the input file. The dependency file
            can be included in a make file for the .c/.cc/.cpp/.cxx/.cu input file. 
            This option cannot be specified together with -E. 
            The dependency file name is computed as follows:
            - If -MF is specified, then the specified file is used as the dependency
            file name 
            - If -o is specified, the dependency file name is computed from the specified
            file name by replacing the suffix with '.d'.
            - Otherwise, the dependency file name is computed by replacing the input
            file names's suffix with '.d'
            If the dependency file name is computed based on either -MF or -o, then multiple
            input files are not supported.
    
    --generate-nonsystem-dependencies-with-compile  (-MMD)                          
            Same as --generate-dependencies-with-compile, but skip header files found
            in system directories (Linux only).
    
    --dependency-output                             (-MF)                           
            Specify the output file for the dependency file generated with -M/-MM/-MD/-MMD.
            
    
    --generate-dependency-targets                   (-MP)                           
            Add an empty target for each dependency.
    
    --compile                                       (-c)                            
            Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file.
    
    --device-c                                      (-dc)                           
            Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains
            relocatable device code.  It is equivalent to '--relocatable-device-code=true
            --compile'.
    
    --device-w                                      (-dw)                           
            Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains
            executable device code.  It is equivalent to '--relocatable-device-code=false
            --compile'.
    
    --device-link                                   (-dlink)                        
            Link object files with relocatable device code and .ptx/.cubin/.fatbin files
            into an object file with executable device code, which can be passed to the
            host linker.
    
    --link                                          (-link)                         
            This option specifies the default behavior: compile and link all inputs.
    
    --lib                                           (-lib)                          
            Compile all inputs into object files (if necessary) and add the results to
            the specified output library file.
    
    --run                                           (-run)                          
            This option compiles and links all inputs into an executable, and executes
            it.  Or, when the input is a single executable, it is executed without any
            compilation or linking. This step is intended for developers who do not want
            to be bothered with setting the necessary environment variables; these are
            set temporarily by nvcc).
    
    
    File and path specifications.
    =============================
    
    --output-file <file>                            (-o)                            
            Specify name and location of the output file.  Only a single input file is
            allowed when this option is present in nvcc non-linking/archiving mode.
    
    --pre-include <file>,...                        (-include)                      
            Specify header files that must be preincluded during preprocessing.
    
    --objdir-as-tempdir                             (-objtemp)                      
            Create intermediate files in the same directory as the object file instead
            of in the temporary directory. This option will take effect only if -c, -dc
            or -dw is also used.
    
    --library <library>,...                         (-l)                            
            Specify libraries to be used in the linking stage without the library file
            extension.  The libraries are searched for on the library search paths that
            have been specified using option '--library-path'.
    
    --define-macro <def>,...                        (-D)                            
            Specify macro definitions to define for use during preprocessing or compilation.
    
    --undefine-macro <def>,...                      (-U)                            
            Undefine macro definitions during preprocessing or compilation.
    
    --include-path <path>,...                       (-I)                            
            Specify include search paths.
    
    --system-include <path>,...                     (-isystem)                      
            Specify system include search paths.
    
    --library-path <path>,...                       (-L)                            
            Specify library search paths.
    
    --output-directory <directory>                  (-odir)                         
            Specify the directory of the output file.  This option is intended for letting
            the dependency generation step (see '--generate-dependencies') generate a
            rule that defines the target object file in the proper directory.
    
    --compiler-bindir <path>                        (-ccbin)                        
            Specify the directory in which the host compiler executable resides.  The
            host compiler executable name can be also specified to ensure that the correct
            host compiler is selected.  In addition, driver prefix options ('--input-drive-prefix',
            '--dependency-drive-prefix', or '--drive-prefix') may need to be specified,
            if nvcc is executed in a Cygwin shell or a MinGW shell on Windows.
    
    --allow-unsupported-compiler                    (-allow-unsupported-compiler)   
            Disable nvcc check for supported host compiler versions. Using an unsupported
            host compiler may cause compilation failure or incorrect run time execution.
            Use at your own risk. This option has no effect on MacOS.
    
    --archiver-binary <path>                        (-arbin)                        
            Specify the path of the executable for the archiving tool used to createstatic
            libraries with '--lib'. If unspecified, a platform-specific defaultis used.
    
    --cudart {none|shared|static}                   (-cudart)                       
            Specify the type of CUDA runtime library to be used: no CUDA runtime library,
            shared/dynamic CUDA runtime library, or static CUDA runtime library.
            Allowed values for this option:  'none','shared','static'.
            Default value:  'static'.
    
    --cudadevrt {none|static}                       (-cudadevrt)                    
            Specify the type of CUDA device runtime library to be used: no CUDA device
            runtime library, or static CUDA device runtime library.
            Allowed values for this option:  'none','static'.
            Default value:  'static'.
    
    --libdevice-directory <directory>               (-ldir)                         
            Specify the directory that contains the libdevice library files when option
            '--dont-use-profile' is used.  Libdevice library files are located in the
            'nvvm/libdevice' directory in the CUDA toolkit.
    
    --target-directory <string>                     (-target-dir)                   
            Specify the subfolder name in the targets directory where the default include
            and library paths are located. 
    
    --use-local-env                                 (-use-local-env)                
            By default nvcc assumes that the MSVC environment needs to be initialized.
            This is done by executing the appropriate command file available for the
            MSVC installation detected or specified. Initializing the environment for
            each nvcc invocation can add noticeable overheads. If the environment used
            to invoke nvcc has already been configured, this option can be used to skip
            this step.
    
    
    Options for specifying behavior of compiler/linker.
    ===================================================
    
    --profile                                       (-pg)                           
            Instrument generated code/executable for use by gprof (Linux only).
    
    --debug                                         (-g)                            
            Generate debug information for host code.
    
    --device-debug                                  (-G)                            
            Generate debug information for device code. If --dopt is not specified, then
            turns off all optimizations. Don't use for profiling; use -lineinfo instead.
    
    --generate-line-info                            (-lineinfo)                     
            Generate line-number information for device code.
    
    --optimization-info <kind>,...                  (-opt-info)                     
            Provide optimization reports for the specified kind of optimization. The
            following tags are supported:
                    inline: Emit remarks related to function inlining. Inlining passmay
            be invoked multiple times by the compiler and a function notinlined in an
            earlier pass may be inlined in a subsequent pass.
            Allowed values for this option:  'inline'.
    
    --optimize <level>                              (-O)                            
            Specify optimization level for host code.
    
    --dopt <kind>                                   (-dopt)                         
            Enable device code optimization. When specified along with '-G', enables
            limited debug information generation for optimized device code (currently,
            only line number information). When '-G' is not specified, '-dopt=on' is
            implicit.
            Allowed values for this option:  'on'.
    
    --dlink-time-opt                                (-dlto)                         
            Perform link-time optimization of device code. Link-time optimization must
            be specified at both compile and link time; at compile time it stores high-level
            intermediate code, then at link time it links together and optimizes the
            intermediate code.If that intermediate is not found at link time then nothing
            happens. Intermediate code is also stored at compile time with the --gpu-code='lto_NN'
            target. The options -dlto -arch=sm_NN will add a lto_NN target; if you want
            to only add a lto_NN target and not the compute_NN that -arch=sm_NN usually
            generates, use -arch=lto_NN. The options '-dlto -dlink -ptx -o <file.ptx>'
            will cause nvlink to generate <file.ptx>. If -o is not used, the file generated
            will be a_dlink.dlto.ptx.
    
    --lto                                           (-lto)                          
            Alias for -dlto.
    
    --ftemplate-backtrace-limit <limit>             (-ftemplate-backtrace-limit)    
            Set the maximum number of template instantiation notes for a single warning
            or error to <limit>. A value of 0 is allowed, and indicates that no limit
            should be enforced. This value is also passed to the host compiler if it
            provides an equivalent flag.
    
    --ftemplate-depth <limit>                       (-ftemplate-depth)              
            Set the maximum instantiation depth for template classes to <limit>. This
            value is also passed to the host compiler if it provides an equivalent flag.
    
    --no-exceptions                                 (-noeh)                         
            Disable exception handling for host code.
    
    --shared                                        (-shared)                       
            Generate a shared library during linking.  Use option '--linker-options'
            when other linker options are required for more control.
    
    --x {c|c++|cu}                                  (-x)                            
            Explicitly specify the language for the input files, rather than letting
            the compiler choose a default based on the file name suffix.
            Allowed values for this option:  'c','c++','cu'.
    
    --std {c++03|c++11|c++14|c++17}                 (-std)                          
            Select a particular C++ dialect.  Note that this flag also turns on the corresponding
            dialect flag for the host compiler.
            Allowed values for this option:  'c++03','c++11','c++14','c++17'.
    
    --no-host-device-initializer-list               (-nohdinitlist)                 
            Do not implicitly consider member functions of std::initializer_list as __host__
            __device__ functions.
    
    --no-host-device-move-forward                   (-nohdmoveforward)              
            Do not implicitly consider std::move and std::forward as __host__ __device__
            function templates.
    
    --expt-relaxed-constexpr                        (-expt-relaxed-constexpr)       
            Experimental flag: Allow host code to invoke __device__ constexpr functions,
            and device code to invoke __host__ constexpr functions.Note that the behavior
            of this flag may change in future compiler releases.
    
    --extended-lambda                               (-extended-lambda)              
            Allow __host__, __device__ annotations in lambda declaration. 
    
    --expt-extended-lambda                          (-expt-extended-lambda)         
            Alias for -extended-lambda.
    
    --machine {32|64}                               (-m)                            
            Specify 32 vs 64 bit architecture.
            Allowed values for this option:  32,64.
            Default value:  64.
    
    --m32                                           (-m32)                          
            Equivalent to --machine=32.
    
    --m64                                           (-m64)                          
            Equivalent to --machine=64.
    
    
    Options for passing specific phase options
    ==========================================
    These allow for passing options directly to the intended compilation phase.  Using
    these, users have the ability to pass options to the lower level compilation tools,
    without the need for nvcc to know about each and every such option.
    
    --compiler-options <options>,...                (-Xcompiler)                    
            Specify options directly to the compiler/preprocessor.
    
    --linker-options <options>,...                  (-Xlinker)                      
            Specify options directly to the host linker.
    
    --archive-options <options>,...                 (-Xarchive)                     
            Specify options directly to library manager.
    
    --ptxas-options <options>,...                   (-Xptxas)                       
            Specify options directly to ptxas, the PTX optimizing assembler.
    
    --nvlink-options <options>,...                  (-Xnvlink)                      
            Specify options directly to nvlink.
    
    
    Miscellaneous options for guiding the compiler driver.
    ======================================================
    
    --forward-unknown-to-host-compiler              (-forward-unknown-to-host-compiler)
            Forward unknown options to the host compiler. An 'unknown option' is a command
            line argument that starts with '-' followed by another character, and is
            not a recognized nvcc flag or an argument for a recognized nvcc flag.
            Note: If the unknown option is followed by a separate command line argument,
            the argument will not be forwarded, unless it begins with the '-' character.
            E.g.
            'nvcc -forward-unknown-to-host-compiler -foo=bar a.cu' will forward '-foo=bar'
            to host compiler.
            'nvcc -forward-unknown-to-host-compiler -foo bar a.cu' will report an error
            for 'bar' argument.
            'nvcc -forward-unknown-to-host-compiler -foo -bar a.cu' will forward '-foo'
            and '-bar' to host compiler.
    
    --forward-unknown-to-host-linker                (-forward-unknown-to-host-linker)
            Forward unknown options to the host linker. An 'unknown option' is a command
            line argument that starts with '-' followed by another character, and is
            not a recognized nvcc flag or an argument for a recognized nvcc flag.
            Note: If the unknown option is followed by a separate command line argument,
            the argument will not be forwarded, unless it begins with the '-' character.
            E.g.
            'nvcc -forward-unknown-to-host-linker -foo=bar a.cu' will forward '-foo=bar'
            to host linker.
            'nvcc -forward-unknown-to-host-linker -foo bar a.cu' will report an error
            for 'bar' argument.
            'nvcc -forward-unknown-to-host-linker -foo -bar a.cu' will forward '-foo'
            and '-bar' to host linker.
    
    --forward-unknown-opts                          (-forward-unknown-opts)         
            Implies the combination of options: -forward-unknown-to-host-linker and -forward-unknown-to-host-compiler.
            E.g.
            'nvcc -forward-unknown-opts -foo=bar a.cu' will forward '-foo=bar' to the
            host linker and compiler.
            'nvcc -forward-unknown-opts -foo bar a.cu' will report an error for 'bar'
            argument.
            'nvcc -forward-unknown-opts -foo -bar a.cu' will forward '-foo' and '-bar'
            to the host linker and compiler.
    
    --dont-use-profile                              (-noprof)                       
            Nvcc uses the nvcc.profiles file for compilation.  When specifying this option,
            the profile file is not used.
    
    --dryrun                                        (-dryrun)                       
            Do not execute the compilation commands generated by nvcc.  Instead, list
            them.
    
    --verbose                                       (-v)                            
            List the compilation commands generated by this compiler driver, but do not
            suppress their execution.
    
    --threads <number>                              (-t)                            
            Specify the maximum number of threads to be created in parallel when compiling
            for multiple architectures. If <number> is 1 or if compiling for one architecture,
            this option is ignored. If <number> is 0, the number of threads will be the
            number of CPUs on the machine.
    
    --keep                                          (-keep)                         
            Keep all intermediate files that are generated during internal compilation
            steps.
    
    --keep-dir <directory>                          (-keep-dir)                     
            Keep all intermediate files that are generated during internal compilation
            steps in this directory.
    
    --save-temps                                    (-save-temps)                   
            This option is an alias of '--keep'.
    
    --clean-targets                                 (-clean)                        
            This option reverses the behavior of nvcc.  When specified, none of the compilation
            phases will be executed.  Instead, all of the non-temporary files that nvcc
            would otherwise create will be deleted.
    
    --time <file name>                              (-time)                         
            Generate a comma separated value table with the time taken by each compilation
            phase, and append it at the end of the file given as the option argument.
            If the file is empty, the column headings are generated in the first row
            of the table. If the file name is '-', the timing data is generated in stdout.
    
    --run-args <arguments>,...                      (-run-args)                     
            Used in combination with option --run to specify command line arguments for
            the executable.
    
    --input-drive-prefix <prefix>                   (-idp)                          
            On Windows, all command line arguments that refer to file names must be converted
            to the Windows native format before they are passed to pure Windows executables.
            This option specifies how the current development environment represents
            absolute paths.  Use '/cygwin/' as <prefix> for Cygwin build environments,
            and '/' as <prefix> for MinGW.
    
    --dependency-drive-prefix <prefix>              (-ddp)                          
            On Windows, when generating dependency files (see --generate-dependencies),
            all file names must be converted appropriately for the instance of 'make'
            that is used.  Some instances of 'make' have trouble with the colon in absolute
            paths in the native Windows format, which depends on the environment in which
            the 'make' instance has been compiled.  Use '/cygwin/' as <prefix> for a
            Cygwin make, and '/' as <prefix> for MinGW.  Or leave these file names in
            the native Windows format by specifying nothing.
    
    --drive-prefix <prefix>                         (-dp)                           
            Specifies <prefix> as both --input-drive-prefix and --dependency-drive-prefix.
    
    --dependency-target-name <target>               (-MT)                           
            Specify the target name of the generated rule when generating a dependency
            file (see '--generate-dependencies').
    
    --no-align-double                               --no-align-double               
            Specifies that '-malign-double' should not be passed as a compiler argument
            on 32-bit platforms.  WARNING: this makes the ABI incompatible with the cuda's
            kernel ABI for certain 64-bit types.
    
    --no-device-link                                (-nodlink)                      
            Skip the device link step when linking object files.
    
    
    Options for steering GPU code generation.
    =========================================
    
    --gpu-architecture <arch>                       (-arch)                         
            Specify the name of the class of NVIDIA 'virtual' GPU architecture for which
            the CUDA input files must be compiled.
            With the exception as described for the shorthand below, the architecture
            specified with this option must be a 'virtual' architecture (such as compute_50).
            Normally, this option alone does not trigger assembly of the generated PTX
            for a 'real' architecture (that is the role of nvcc option '--gpu-code',
            see below); rather, its purpose is to control preprocessing and compilation
            of the input to PTX.
            For convenience, in case of simple nvcc compilations, the following shorthand
            is supported.  If no value for option '--gpu-code' is specified, then the
            value of this option defaults to the value of '--gpu-architecture'.  In this
            situation, as only exception to the description above, the value specified
            for '--gpu-architecture' may be a 'real' architecture (such as a sm_50),
            in which case nvcc uses the specified 'real' architecture and its closest
            'virtual' architecture as effective architecture values.  For example, 'nvcc
            --gpu-architecture=sm_50' is equivalent to 'nvcc --gpu-architecture=compute_50
            --gpu-code=sm_50,compute_50'.
            -arch=all         build for all supported architectures (sm_*), and add PTX
            for the highest major architecture to the generated code.
            -arch=all-major   build for just supported major versions (sm_*0), plus the
            earliest supported, and add PTX for the highest major architecture to the
            generated code.
            -arch=native      build for all architectures (sm_*) on the current system
            Note: -arch=native, -arch=all, -arch=all-major cannot be used with the -code
            option, but can be used with -gencode options
            Note: the values compute_30, compute_32, compute_35, compute_37, compute_50,
            sm_30, sm_32, sm_35, sm_37 and sm_50 are deprecated and may be removed in
            a future release.
            Allowed values for this option:  'all','all-major','compute_35','compute_37',
            'compute_50','compute_52','compute_53','compute_60','compute_61','compute_62',
            'compute_70','compute_72','compute_75','compute_80','compute_86','compute_87',
            'lto_35','lto_37','lto_50','lto_52','lto_53','lto_60','lto_61','lto_62',
            'lto_70','lto_72','lto_75','lto_80','lto_86','lto_87','native','sm_35','sm_37',
            'sm_50','sm_52','sm_53','sm_60','sm_61','sm_62','sm_70','sm_72','sm_75',
            'sm_80','sm_86','sm_87'.
    
    --gpu-code <code>,...                           (-code)                         
            Specify the name of the NVIDIA GPU to assemble and optimize PTX for.
            nvcc embeds a compiled code image in the resulting executable for each specified
            <code> architecture, which is a true binary load image for each 'real' architecture
            (such as sm_50), and PTX code for the 'virtual' architecture (such as compute_50).
            During runtime, such embedded PTX code is dynamically compiled by the CUDA
            runtime system if no binary load image is found for the 'current' GPU.
            Architectures specified for options '--gpu-architecture' and '--gpu-code'
            may be 'virtual' as well as 'real', but the <code> architectures must be
            compatible with the <arch> architecture.  When the '--gpu-code' option is
            used, the value for the '--gpu-architecture' option must be a 'virtual' PTX
            architecture.
            For instance, '--gpu-architecture=compute_60' is not compatible with '--gpu-code=sm_52',
            because the earlier compilation stages will assume the availability of 'compute_60'
            features that are not present on 'sm_52'.
            Note: the values compute_30, compute_32, compute_35, compute_37, compute_50,
            sm_30, sm_32, sm_35, sm_37 and sm_50 are deprecated and may be removed in
            a future release.
            Allowed values for this option:  'compute_35','compute_37','compute_50',
            'compute_52','compute_53','compute_60','compute_61','compute_62','compute_70',
            'compute_72','compute_75','compute_80','compute_86','compute_87','lto_35',
            'lto_37','lto_50','lto_52','lto_53','lto_60','lto_61','lto_62','lto_70',
            'lto_72','lto_75','lto_80','lto_86','lto_87','sm_35','sm_37','sm_50','sm_52',
            'sm_53','sm_60','sm_61','sm_62','sm_70','sm_72','sm_75','sm_80','sm_86',
            'sm_87'.
    
    --generate-code <specification>,...             (-gencode)                      
            This option provides a generalization of the '--gpu-architecture=<arch> --gpu-code=<code>,
            ...' option combination for specifying nvcc behavior with respect to code
            generation.  Where use of the previous options generates code for different
            'real' architectures with the PTX for the same 'virtual' architecture, option
            '--generate-code' allows multiple PTX generations for different 'virtual'
            architectures.  In fact, '--gpu-architecture=<arch> --gpu-code=<code>,
            ...' is equivalent to '--generate-code arch=<arch>,code=<code>,...'.
            '--generate-code' options may be repeated for different virtual architectures.
            Allowed keywords for this option:  'arch','code'.
    
    --relocatable-device-code {true|false}          (-rdc)                          
            Enable (disable) the generation of relocatable device code.  If disabled,
            executable device code is generated.  Relocatable device code must be linked
            before it can be executed.
            Default value:  false.
    
    --entries entry,...                             (-e)                            
            Specify the global entry functions for which code must be generated.  By
            default, code will be generated for all entry functions.
    
    --maxrregcount <amount>                         (-maxrregcount)                 
            Specify the maximum amount of registers that GPU functions can use.
            Until a function-specific limit, a higher value will generally increase the
            performance of individual GPU threads that execute this function.  However,
            because thread registers are allocated from a global register pool on each
            GPU, a higher value of this option will also reduce the maximum thread block
            size, thereby reducing the amount of thread parallelism.  Hence, a good maxrregcount
            value is the result of a trade-off.
            If this option is not specified, then no maximum is assumed.
            Value less than the minimum registers required by ABI will be bumped up by
            the compiler to ABI minimum limit.
            User program may not be able to make use of all registers as some registers
            are reserved by compiler.
    
    --use_fast_math                                 (-use_fast_math)                
            Make use of fast math library.  '--use_fast_math' implies '--ftz=true --prec-div=false
            --prec-sqrt=false --fmad=true'.
    
    --ftz {true|false}                              (-ftz)                          
            This option controls single-precision denormals support. '--ftz=true' flushes
            denormal values to zero and '--ftz=false' preserves denormal values. '--use_fast_math'
            implies '--ftz=true'.
            Default value:  false.
    
    --prec-div {true|false}                         (-prec-div)                     
            This option controls single-precision floating-point division and reciprocals.
            '--prec-div=true' enables the IEEE round-to-nearest mode and '--prec-div=false'
            enables the fast approximation mode.  '--use_fast_math' implies '--prec-div=false'.
            Default value:  true.
    
    --prec-sqrt {true|false}                        (-prec-sqrt)                    
            This option controls single-precision floating-point squre root.  '--prec-sqrt=true'
            enables the IEEE round-to-nearest mode and '--prec-sqrt=false' enables the
            fast approximation mode.  '--use_fast_math' implies '--prec-sqrt=false'.
            Default value:  true.
    
    --fmad {true|false}                             (-fmad)                         
            This option enables (disables) the contraction of floating-point multiplies
            and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA,
            or DFMA).  '--use_fast_math' implies '--fmad=true'.
            Default value:  true.
    
    --extra-device-vectorization                    (-extra-device-vectorization)   
            This option enables more aggressive device code vectorization.
    
    
    Options for steering cuda compilation.
    ======================================
    
    --default-stream {legacy|null|per-thread}       (-default-stream)               
            Specify the stream that CUDA commands from the compiled program will be sent
            to by default.
                    
            legacy
                    The CUDA legacy stream (per context, implicitly synchronizes with
                    other streams).
                    
            per-thread
                    A normal CUDA stream (per thread, does not implicitly
                    synchronize with other streams).
                    
            'null' is a deprecated alias for 'legacy'.
                    
            Allowed values for this option:  'legacy','null','per-thread'.
            Default value:  'legacy'.
    
    
    Generic tool options.
    =====================
    
    --disable-warnings                              (-w)                            
            Inhibit all warning messages.
    
    --keep-device-functions                         (-keep-device-functions)        
            In whole program compilation mode, preserve user defined external linkage
            __device__ function definitions up to PTX.
    
    --source-in-ptx                                 (-src-in-ptx)                   
            Interleave source in PTX. May only be used in conjunction with --device-debug
            or --generate-line-info.
    
    --restrict                                      (-restrict)                     
            Programmer assertion that all kernel pointer parameters are restrict pointers.
    
    --Wreorder                                      (-Wreorder)                     
            Generate warnings when member initializers are reordered.
    
    --Wdefault-stream-launch                        (-Wdefault-stream-launch)       
            Generate warning when an explicit stream argument is not provided in the
            <<<...>>> kernel launch syntax.
    
    --Wmissing-launch-bounds                        (-Wmissing-launch-bounds)       
            Generate warning when a __global__ function does not have an explicit __launch_bounds__
            annotation.
    
    --Wext-lambda-captures-this                     (-Wext-lambda-captures-this)    
            Generate warning when an extended lambda implicitly captures 'this'.
    
    --Wno-deprecated-declarations                   (-Wno-deprecated-declarations)  
            Suppress warning on use of deprecated entity.
    
    --Wno-deprecated-gpu-targets                    (-Wno-deprecated-gpu-targets)   
            Suppress warnings about deprecated GPU target architectures.
    
    --Werror <kind>,...                             (-Werror)                       
            Make warnings of the specified kinds into errors.  The following is the list
            of warning kinds accepted by this option:
                    
            cross-execution-space-call
                    Be more strict about unsupported cross execution space calls.
                    The compiler will generate an error instead of a warning for a
                    call from a __host__ __device__ to a __host__ function.
            reorder
                    Generate errors when member initializers are reordered.
            deprecated-declarations
                    Generate error on use of a deprecated entity.
            default-stream-launch
                    Generate error when an explicit stream argument is not provided in
            the <<<...>>> kernel launch syntax.
            missing-launch-bounds
                    Generate error when a __global__ function does not have an explicit
            __launch_bounds__ annotation.
            ext-lambda-captures-this
                    Generate error when an extended lambda implicitly captures 'this'
            Allowed values for this option:  'all-warnings','cross-execution-space-call',
            'default-stream-launch','deprecated-declarations','ext-lambda-captures-this',
            'missing-launch-bounds','reorder'.
    
    --resource-usage                                (-res-usage)                    
            Show resource usage such as registers and memory of the GPU code.
            This option implies '--nvlink-options --verbose' when '--relocatable-device-code=true'
            is set.  Otherwise, it implies '--ptxas-options --verbose'.
    
    --extensible-whole-program                      (-ewp)                          
            Do extensible whole program compilation of device code.
    
    --no-compress                                   (-no-compress)                  
            Do not compress device code in fatbinary.
    
    --qpp-config                                    (-qpp-config)                   
            Specify the configuration ([[compiler/]version,][target]) for the q++ host
            compiler. The argument will be forwarded to the q++ compiler with its -V
            flag.
    
    --compile-as-tools-patch                        (-astoolspatch)                 
            Compile patch code for CUDA tools. Implies --keep-device-functions.
    
    --list-gpu-code                                 (-code-ls)                      
            List the gpu architectures (sm_XX) supported by the compiler and exit. If
            both --list-gpu-code and --list-gpu-arch are set, the list is displayed using
            the same format as the --generate-code value.
    
    --list-gpu-arch                                 (-arch-ls)                      
            List the virtual device architectures (compute_XX) supported by the compiler
            and exit. If both --list-gpu-code and --list-gpu-arch are set, the list is
            displayed using the same format as the --generate-code value.
    
    --version-ident {true|false}                    (-dQ)                           
            This option enables (disables) the generation of compiler tool version identity
            in device code object.
            Default value:  false.
    
    --display-error-number                          (-err-no)                       
            This option displays a diagnostic number for any message generated by the
            CUDA frontend compiler (note: not the host compiler).
    
    --no-display-error-number                       (-no-err-no)                    
            This option disables the display of a diagnostic number for any message generated
            by the CUDA frontend compiler (note: not the host compiler).
    
    --diag-error <error-number>,...                 (-diag-error)                   
            Emit error for specified diagnostic message(s) generated by the CUDA frontend
            compiler (note: does not affect diagnostics generated by the host compiler/preprocessor).
    
    --diag-suppress <error-number>,...              (-diag-suppress)                
            Suppress specified diagnostic message(s) generated by the CUDA frontend compiler
            (note: does not affect diagnostics generated by the host compiler/preprocessor).
    
    --diag-warn <error-number>,...                  (-diag-warn)                    
            Emit warning for specified diagnostic message(s) generated by the CUDA frontend
            compiler (note: does not affect diagnostics generated by the host compiler/preprocessor).
    
    --host-linker-script {use-lcs|gen-lcs}          (-hls)                          
            Use the host linker script (GNU/Linux only) to enable support for certain
            CUDA specific requirements, while building executable files or shared libraries.
                    
            use-lcs
                    Prepares a host linker script to enable host linker support 
                    for relocatable device object files that are larger in size,
                    that would otherwise, in certain cases cause the host
                    linker to fail with relocation truncation error.
            gen-lcs
                    Generates a host linker script that can be passed to host 
                    linker manually, in the case where host linker is invoked 
                    separately outside of nvcc. This option can be combined 
                    with -shared or -r option to generate linker scripts that 
                    can be used while generating host shared libraries or host 
                    relocatable links respectively.
                    
                    The file generated using this option must be provided as 
                    the last input file to the host linker.
                    
                    Default Output Filename: The output is generated to stdout 
                    by default. Use the option -o filename to specify the 
                    output filename.
                    
            A linker script may already be in use and passed 
            to the host linker using the host linker option --script 
            (or -T), then the generated host linker script must augment 
            the existing linker script. In such cases, the option -aug-hls 
            must be used to generate linker script that contains only the 
            augmentation parts. Otherwise, the host linker behaviour is 
            undefined.
                    
            A host linker option, such as -z with a non-default argument, 
            that can modify the default linker script internally, is 
            incompatible with this option and the behavior of any such 
            usage is undefined.
                    
            Allowed values for this option:  'gen-lcs','use-lcs'.
    
    --augment-host-linker-script                    (-aug-hls)                      
            Enables generation of host linker script that augments an existing host linker
            script (GNU/Linux only). See option --host-linker-script for more details.
    
    --host-relocatable-link                         (-r)                            
            When used in combination with -hls=gen-lcs, controls the behaviour of -hls=gen-lcs
            and sets it to generate host linker script that can be used in host relocatable
            link (ld -r linkage). See option -hls=gen-lcs for more information.
                    
            This option currently is effective only when used with -hls=gen-lcs; in all
            other cases, this option is ignored currently.
    
    --help                                          (-h)                            
            Print this help information on this tool.
    
    --version                                       (-V)                            
            Print version information on this tool.
    
    --options-file <file>,...                       (-optf)                         
            Include command line options from specified file.
    
    
    
    ```

* nvcc和g++有许多相似可以共用的指令，可以对比查看。

##### （2）Makefile

* (.cu)文件下的Makefile：

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/13076fb7-3e06-445d-948a-56f81ca90a74)


  * ```python
    # 可以包含其他目录下的文件
    CONFIG        :=  ../../config/Makefile.config
    CONFIG_LOCAL  :=  ./config/Makefile.config
    
    include $(CONFIG)
    include $(CONFIG_LOCAL)
    
    # 变量的定义
    BUILD_PATH    :=  build
    SRC_PATH      :=  src
    CUDA_DIR      :=  /usr/local/cuda-$(CUDA_VER)
    
    # wildcard用法：$(wildcard $(SRC_PATH)/*.cu)寻找SRC_PATH目录下的所有(.cu)文件
    KERNELS_SRC   :=  $(wildcard $(SRC_PATH)/*.cu)
    
    # patsubst用法：$(patsubst  <pattern>,<replacement>,<text>)替换文件后缀
    # patsubst函数返回被替换过后的字符串。
    # patsubst函数判断<text>中字符串（若多个字符串以空格分隔）是否匹配<pattern>模式，
    # 若匹配则使用<replacement>替换<text>。
    APP_OBJS      +=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  
    
    APP_DEPS      +=  $(KERNELS_SRC)
    
    CUCC          :=  $(CUDA_DIR)/bin/nvcc
    
    CUDAFLAGS     :=  -Xcompiler -fPIC 
    
    INCS          :=  -I $(CUDA_DIR)/include \
                      -I $(SRC_PATH) 
    
    LIBS          :=  -L "$(CUDA_DIR)/lib64" \
    
    # DEBUG：nvcc要进行DEBUG要把-g -G放进来
    # -O0：不做任何优化，-OC：做优化
    ifeq ($(DEBUG),1)
    CUDAFLAGS     +=  -g -G -O0
    else
    CUDAFLAGS     +=  -O3
    endif
    
    # 打印WARNING
    ifeq ($(SHOW_WARNING),1)
    # 打印警告
    CUDAFLAGS     +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
    else
    # 不想打印警告
    CUDAFLAGS     +=  -w
    endif
    
    # Makefile依赖关系部分
    
    # MAKE：makefile自己内部定义的变量
    # APP：trt-cuda可执行文件
    all:
    	$(MAKE) $(APP)
    
    update: $(APP)
    	@echo finished updating $<
    
    # 设置APP依赖关系（是在.o文件生成之后）是APP_DEPS和APP_OBJS
    $(APP): $(APP_DEPS) $(APP_OBJS)
    # nvcc和所有的.o文件链接在一起
    # 输出生成(-o)CUDA的可执行文件(trt-cuda)
    # 并且加上$(LIBS) $(INCS)这些连接库
    	@$(CUCC) $(APP_OBJS) -o $@ $(LIBS) $(INCS)
    	@echo finished building $@. Have fun!!
    
    show: 
    	@echo $(BUILD_PATH)
    	@echo $(APP_DEPS)
    	@echo $(INCS)
    	@echo $(APP_OBJS)
    
    clean:
    	rm -rf $(APP)
    	rm -rf build
    
    # Compile CUDA
    # 对于BUILD_PATH目录下的(.cu.o)文件依赖SRC_PATH目录下的(.cu)文件
    $(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
    	@echo Compile CUDA $@
    	@mkdir -p $(BUILD_PATH)
    	@$(CUCC) -o $@ -c $< $(CUDAFLAGS) $(INCS)
    
    .PHONY: all update show clean 
    
    ```

* (.cpp)文件下的Makefile：

  * 在src文件夹下多了一个【main.cpp】、【print_index.hpp】、【utils.hpp】

  * 意味着可以以【main.cpp】为接口，用c++编译器去编译它，从它里面调用print_index函数

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d62cd926-1fe6-40ea-aba2-639fee68d61a)


    * main.cpp中不会涉及跟cuda相关的函数，但是会有相关函数的接口（比如print_one_dim）：

      * 我们不能够从main.cpp中直接调用cuda语法<<<>>>，<<<>>>是nvcc(cuda编译器)才可以识别的语法，g++不可以

      * ```c++
        #include <stdio.h>
        #include <cuda_runtime.h>
        #include "print_index.hpp"
        
        void print_one_dim(int inputSize, int blockSize){
            int gridSize = inputSize / blockSize;
        
            dim3 block(blockSize);
            dim3 grid(gridSize);
        
            // print_idx_device(block, grid);
            // print_dim_device(block, grid);
            // print_thread_idx_per_block_device(block, grid);
            print_thread_idx_device(block, grid);
        }
        
        void print_two_dim(int inputSize, int blockSize){
            int gridSize = inputSize / blockSize;
        
            dim3 block(blockSize, blockSize);
            dim3 grid(gridSize, gridSize);
        
            // print_idx_device(block, grid);
            // print_dim_device(block, grid);
            // print_thread_idx_per_block_device(block, grid);
            print_thread_idx_device(block, grid);
        }
        
        
        int main(){
            int inputSize;
            int blockSize;
        
            /* one-dimention test */
            // inputSize = 32;
            // blockSize = 4;
            // print_one_dim(inputSize, blockSize);
                
            /* two-dimention test */
            inputSize = 8;
            blockSize = 4;
            print_two_dim(inputSize, blockSize);
            return 0;
        }
        ```

    * .cu中提供给.cpp的接口函数定义在了.hpp中，这样可以在make的时候通过头文件找到函数的声明

      * ```c++
        #ifndef __PRINT_INDEX_HPP
        #define __PRINT_INDEX_HPP
        
        #include <cuda_runtime.h>
        void print_idx_device(dim3 grid, dim3 block);
        void print_dim_device(dim3 grid, dim3 block);
        void print_thread_idx_per_block_device(dim3 grid, dim3 block);
        void print_thread_idx_device(dim3 grid, dim3 block);
        
        #endif //__PRINT_INDEX_HPP
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3755465a-afb1-46c9-8e6e-606411658ff2)


    * CUDA_CHECK是一种CUDA程序的排查错误的手段（error handler）。

      * ```c++
        #ifndef __UTILS_HPP__
        #define __UTILS_HPP__
        
        #include <cuda_runtime.h>
        #include <system_error>
        
        // 一般cuda的check都是这样写成宏
        #define CUDA_CHECK(call) {                                                 \
            cudaError_t error = call;                                              \
            if (error != cudaSuccess) {                                            \
                printf("ERROR: %s:%d, ", __FILE__, __LINE__);                      \
                printf("CODE:%d, DETAIL:%s\n", error, cudaGetErrorString(error));  \
                exit(1);                                                           \
            }                                                                      \
        }
        
        #endif //__UTILS__HPP__
        ```

      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a34f1d88-46a0-4e3f-ae02-b76f746910ed)


  * ```python
    CONFIG        :=  ../../config/Makefile.config
    CONFIG_LOCAL  :=  ./config/Makefile.config
    
    include $(CONFIG)
    include $(CONFIG_LOCAL)
    
    BUILD_PATH    :=  build
    SRC_PATH      :=  src
    CUDA_DIR      :=  /usr/local/cuda-$(CUDA_VER)
    
    # 加入cpp的SRC路径
    CXX_SRC       +=  $(wildcard $(SRC_PATH)/*.cpp)
    KERNELS_SRC   :=  $(wildcard $(SRC_PATH)/*.cu)
    
    # 加入cpp的APP_OBJS路径
    APP_OBJS      :=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(CXX_SRC:.cpp=.cpp.o))
    APP_OBJS      +=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  
    
    # 加入cpp的APP_DEPS路径
    APP_DEPS      :=  $(CXX_SRC)
    APP_DEPS      +=  $(KERNELS_SRC)
    # 加入头文件
    APP_DEPS      +=  $(wildcard $(SRC_PATH)/*.hpp)
    
    
    CUCC          :=  $(CUDA_DIR)/bin/nvcc
    # 加入cpp的compiler的选项
    CXXFLAGS      :=  -std=c++11 -fPIC
    CUDAFLAGS     :=  -Xcompiler -fPIC 
    
    INCS          :=  -I $(CUDA_DIR)/include \
                      -I $(SRC_PATH) 
    
    LIBS          :=  -L "$(CUDA_DIR)/lib64" \
    
    ifeq ($(DEBUG),1)
    CUDAFLAGS     +=  -g -O0 -G
    # cpp的DEBUG不需要-G
    CXXFLAGS      +=  -g -O0
    else
    CUDAFLAGS     +=  -O3
    CXXFLAGS      +=  -O3
    endif
    
    ifeq ($(SHOW_WARNING),1)
    CUDAFLAGS     +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
    CXXFLAGS      +=  -Wall -Wunused-function -Wunused-variable -Wfatal-errors
    else
    CUDAFLAGS     +=  -w
    CXXFLAGS      +=  -w
    endif
    
    all:
    	$(MAKE) $(APP)
    
    update: $(APP)
    	@echo finished updating $<
    
    $(APP): $(APP_DEPS) $(APP_OBJS)
    	@$(CUCC) $(APP_OBJS) -o $@ $(LIBS) $(INCS)
    	@echo finished building $@. Have fun!!
    
    show: 
    	@echo $(BUILD_PATH)
    	@echo $(APP_DEPS)
    	@echo $(INCS)
    	@echo $(APP_OBJS)
    	@echo $(APP_MKS)
    
    clean:
    	rm -rf $(APP)
    	rm -rf build
    
    ifneq ($(MAKECMDGOALS), clean)
    -include $(APP_MKS)
    endif
    
    # Compile CXX
    $(BUILD_PATH)/%.cpp.o: $(SRC_PATH)/%.cpp
    	@echo Compile CXX $@
    	@mkdir -p $(BUILD_PATH)
    # CC就是g++
    	@$(CC) -o $@ -c $< $(CXXFLAGS) $(INCS)
    
    # Compile CUDA
    $(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.cu
    	@echo Compile CUDA $@
    	@mkdir -p $(BUILD_PATH)
    # CUCC就是nvcc
    	@$(CUCC) $(ARCH) -o $@ -c $< $(CUDAFLAGS) $(INCS)
    
    .PHONY: all update show clean 
    ```

### 2.2 使用CUDA进行MATMULL计算

#### 2.2.1 host端与device端的数据传输

##### （1）CPU(host)端

* 1、分配host与device端的内存空间：
  * ![image-20240410154341775](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240410154341775.png)
  * cudaMalloc (在device端分配空间)，是一种cuda runtime api(*)
    * ![image-20240410103035827](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240410103035827.png)
    * (*)这些以cuda*开头的api一般被称作**cuda runtime api**，以CU*开头的api被称作**cuda  driver api**。
      * **cuda runtime api**对底层的操作做好的封装便于使用，包括：
        * implicit initialization (隐式初始化) 
        * context management (上下文管理)
        * module management (模块管理)
      * **cuda  driver api**一般对GPU硬件进行操作，复杂不常用
  * cudaMallocHost (在host端的pinned memory上分配空间)
* 2、将数据传送到GPU
  * ![image-20240410154430491](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240410154430491.png)
  * cudaMemcpy (以同步的方式，将数据在host->device, device->device, device->host进行传输)
  * cudaMemcpyAsync (以异步的方式，进行数据传输)
* 3、配置核函数的参数
  * ![image-20240410154721285](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240410154721285.png)
  * grid dim（必须配置）
  * block dim（必须配置）
  * shared memory size（默认）
  * stream（默认）
* 4、启动核函数
  * 一般是异步的，所以启动完核函数需要进行同步
    * ![image-20240410154750136](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240410154750136.png)
* 5、将数据从GPU传入回来

##### （2） GPU(device)端

* 1、根据配置的参数启动核函数
* 2、多个thread并行计算



#### 2.2.2 CUDA Core的矩阵乘法计算

##### （1）Blocksize=1

![image-20240410103902258](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240410103902258.png)

* 要如果所有的thread依次执行的话：
  * 要完成 4 x 8 (A)与  8 x 4 (B)的计算
  * 需要 8 * 16  = 128个 clk 才可以完成
* 要如果我们分配 16个thread，每一个thread负责C中的一个元素，所有的thread依次执行的话：
  * 要完成 4 x 8 (A)与  8 x 4 (B)的计算 和 一个 1 x 8 与 8 x 1的计算所需要的时间是一样的。
  * 需要 8 个 clk 可以完成。

##### （2）Blocksize=4

![image-20240410111726663](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240410111726663.png)

* 【CUDA中有个规定】：
  * 就是一个block中可以分配的 thread的数量最大是1,024个线程。如果大于 1,024会显示配置错误



#### 2.2.3 代码详情

##### （1）main.cpp

* ```c++
  #include <stdio.h>
  #include <cuda_runtime.h>
  
  #include "utils.hpp"
  #include "timer.hpp"
  #include "matmul.hpp"
  
  
  int seed;
  int main(){
      Timer timer;           // 使用计时器的类
      int width     = 1<<10; // 1,024
      int min       = 0;
      int max       = 1;
      int size      = width * width;
      int blockSize = 1;
  
      float* h_matM = (float*)malloc(size * sizeof(float));
      float* h_matN = (float*)malloc(size * sizeof(float));
      float* h_matP = (float*)malloc(size * sizeof(float));
      float* d_matP = (float*)malloc(size * sizeof(float));
      
      /* 生成矩阵A和B，seed控制生成两个不同矩阵 */
      seed = 1;
      initMatrix(h_matM, size, min, max, seed);   // 矩阵初始化，h_matM：矩阵指针
      seed += 1;
      initMatrix(h_matN, size, min, max, seed);
      
      /* CPU */
      timer.start();
      MatmulOnHost(h_matM, h_matN, h_matP, width);    // A x B = C 的实现，三层for循环
      timer.stop();
      timer.duration<Timer::ms>("matmul in cpu");
  
      /* GPU warmup */
      /* 
       * GPU在第一次启动核函数API的时候存在延迟，影响测量核函数的时间，造成误差
       * 因此需要warmup让CPU第一次执行kernel时随便做点任务，再执行真正想要的内容
       */
      timer.start();
      MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
      timer.stop();
      timer.duration<Timer::ms>("matmul in gpu(warmup)");
  
      /* GPU general implementation, bs = 16*/
      blockSize = 16;
      timer.start();
      MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
      timer.stop();
      timer.duration<Timer::ms>("matmul in gpu(bs = 16)");
      compareMat(h_matP, d_matP, size);   // 确保精度
  
      /* GPU general implementation, bs = 1*/
      blockSize = 1;
      timer.start();
      MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
      timer.stop();
      timer.duration<Timer::ms>("matmul in gpu(bs = 1)");
      compareMat(h_matP, d_matP, size);
  
      /* GPU general implementation, bs = 32*/
      blockSize = 32;
      timer.start();
      MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
      timer.stop();
      timer.duration<Timer::ms>("matmul in gpu(bs = 32)");
      compareMat(h_matP, d_matP, size);
      return 0;
  }
  ```

* ![image-20240410154935349](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240410154935349.png)

#### 2.2.4 CUDA中的Error Handler

Error Handler能帮我们打印出CUDA程序运行中出现的错误，方便我们进行调试



#### 2.2.5 GPU的硬件信息获取

### 2.3 共享内存以及BANK CONFLICT



### 2.4 使用CUDA进行预处理/后处理



### 2.5 STREAM和EVENT
