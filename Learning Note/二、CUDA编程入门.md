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
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2c7e2fe7-5e6b-4f61-8f16-ab86c7f1d299)

  * cudaMalloc (在device端分配空间)，是一种cuda runtime api(*)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/23027c63-1888-4353-ae1e-f7d15091a414)

    * (*)这些以cuda*开头的api一般被称作**cuda runtime api**，以CU*开头的api被称作**cuda  driver api**。
      * **cuda runtime api**对底层的操作做好的封装便于使用，包括：
        * implicit initialization (隐式初始化) 
        * context management (上下文管理)
        * module management (模块管理)
      * **cuda  driver api**一般对GPU硬件进行操作，复杂不常用
  * cudaMallocHost (在host端的pinned memory上分配空间)
* 2、将数据传送到GPU
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/73cf13f7-6fe1-4435-96af-c50c3c30edd3)

  * cudaMemcpy (以同步的方式，将数据在host->device, device->device, device->host进行传输)
  * cudaMemcpyAsync (以异步的方式，进行数据传输)
* 3、配置核函数的参数
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e965cbe8-0bbb-421e-a550-bba5e5add937)

  * grid dim（必须配置）
  * block dim（必须配置）
  * shared memory size（默认）
  * stream（默认）
* 4、启动核函数
  * 一般是异步的，所以启动完核函数需要进行同步
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/66321873-9da7-4752-bdeb-699cf7e752ae)

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

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/68e44d4f-84c5-499a-a51a-01928ff3675a)


#### 2.2.4 CUDA中的Error Handler

Error Handler能帮我们打印出CUDA程序运行中出现的错误，方便我们进行调试

##### （1）新添加的内容

* utils.hpp

  * 添加了两个宏定义

    * ```c++
      // 对cuda runtime api使用的error handling
      inline static void __cudaCheck(cudaError_t err, const char* file, const int line) {
          if (err != cudaSuccess) {
              printf("ERROR: %s:%d, ", file, line);
              printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
              exit(1);
          }
      }
      
      // 对最近的核函数的error handling
      inline static void __kernelCheck(const char* file, const int line) {
          /* 
           * 在编写CUDA是，错误排查非常重要，默认的cuda runtime API中的函数都会返回cudaError_t类型的结果，
           * 但是在写kernel函数的时候，需要通过cudaPeekAtLastError或者cudaGetLastError来获取错误
           */
          cudaError_t err = cudaPeekAtLastError();
          if (err != cudaSuccess) {
              printf("ERROR: %s:%d, ", file, line);
              printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
              exit(1);
          }
      }
      ```

* matmul_gpu_basic.cu

  * 对于每一个cuda runtime api的时候都 进行了错误检查，以及kernel执行结束 以后对kernel进行错误排查

    * ```c++
      void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize){
          /* 设置矩阵大小 */
          int size = width * width * sizeof(float);
      
          /* 分配M, N在GPU上的空间*/
          float *M_device;
          float *N_device;
          CUDA_CHECK(cudaMalloc(&M_device, size));
          CUDA_CHECK(cudaMalloc(&N_device, size));
      
          /* 分配M, N拷贝到GPU上*/ CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));
      
          /* 分配P在GPU上的空间*/
          float *P_device;
          CUDA_CHECK(cudaMalloc(&P_device, size));
      
          /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
          dim3 dimBlock(blockSize, blockSize);
          dim3 dimGrid(width / blockSize, width / blockSize);
          MatmulKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, width);
      
          /* 将结果从device拷贝回host*/
          CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaDeviceSynchronize());
      
          /* 注意要在synchronization结束之后排查kernel的错误, 否则错误排查只会检查参数配置*/
          LAST_KERNEL_CHECK(); 
      
          /* Free */
          cudaFree(P_device);
          cudaFree(N_device);
          cudaFree(M_device);
      }
      ```

##### （2）有无error handling对比

* 没有error handling：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/40bf5f68-c335-4d40-8eab-66df9943808e)

* 有error handling：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b8b2acd8-ae83-4a4d-8f3d-9134c3b7af28)


##### （3）用error handler原因

* 一般来说，我们习惯把cuda的error handler定义成宏。因为这样可以避免在执行的时候发生调用error handler 而引起的overhead。

  * ____FILE____: 编译器内部定义的一个宏。表示的是当前文件的文件名 
  * ____LINE____: 编译器内部定义的一个宏。表示的是当前文件的行

* 一个良好的cuda编程习惯里，我们习惯在调用一个cuda runtime api时，我们就用error handler进行包装。这样可以方便我们排查错误的来源

* ```c++
  #ifndef __UTILS_HPP__
  #define __UTILS_HPP__
  
  #include <cuda_runtime.h>
  #include <system_error>
  
  #define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
  #define LAST_KERNEL_CHECK()          __kernelCheck(__FILE__, __LINE__)
  #define BLOCKSIZE 16
  
  // 对cuda runtime api使用的error handling
  inline static void __cudaCheck(cudaError_t err, const char* file, const int line) {
      if (err != cudaSuccess) {
          printf("ERROR: %s:%d, ", file, line);
          printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
          exit(1);
      }
  }
  
  // 对最近的核函数的error handling
  inline static void __kernelCheck(const char* file, const int line) {
      /* 
       * 在编写CUDA是，错误排查非常重要，默认的cuda runtime API中的函数都会返回cudaError_t类型的结果，
       * 但是在写kernel函数的时候，需要通过cudaPeekAtLastError或者cudaGetLastError来获取错误
       */
      cudaError_t err = cudaPeekAtLastError();
      if (err != cudaSuccess) {
          printf("ERROR: %s:%d, ", file, line);
          printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
          exit(1);
      }
  }
  
  void initMatrix(float* data, int size, int low, int high, int seed);
  void printMat(float* data, int size);
  void compareMat(float* h_data, float* d_data, int size);
  
  #endif //__UTILS_HPP__//
  ```



##### （4）对cuda runtime api使用的error handling

* 在每一个cuda程序执行的时候都会返回cudaError状态指令，而CUDA_CHECK函数便能够打印出来这写状态。
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c24a5b1e-57c9-4c1d-8901-ee3d9c44540f)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/95bb345f-950a-45be-b68a-2ab982f217ab)




##### （5）对核函数的error handling

* 由于核函数返回的类型为空，不像cuda函数能够返回状态信息：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8ef90a8c-eb7d-40cf-b7a4-b437d571d2ac)

* 所以需要特定的捕捉状态函数：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/65716501-bba2-4b1e-94f6-111c66c7aa3e)

  * 这里用到了“cudaError_t err = cudaPeekAtLastError();”指令：
    *![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6ec0aee0-051d-413d-8361-55554b358e1c)

* 在编写CUDA是，错误排查非常重要，默认的cuda runtime API中的函数都会返回**cudaError_t**类型的结果，但是在写kernel函数的时候，需要通过**cudaPeekAtLastError**或者**cudaGetLastError**来获取错误，这两个有本质上的区别。
  * cudaGetLastError：
    * 返回最近的错误，同时把系统状态复位到cudaSuccess状态
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/20a5223e-3361-44f1-9449-170f9342ff92)

  * cudaPeekAtLastError：
    * 返回最近的错误，不会把系统状态复位到cudaSuccess状态
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/02d83b45-0c54-4991-8682-d4720a9679bf)

  * 两者差别在于错误是否传播。
    * 对于不可恢复的错误(*)，如果发生了错误的话并且不把系统的状态进行reset的话，错误会一直传播下去，导致后面的即便正确的api使用也会产生同样的错误。
      *  (*)“不可恢复的(non-recoverable/sticky)”，一般指的是核函数内部的执行错误，比较典型的例子就是内存访问越界。
      * 相比之下，比如像block size以及shared memory size这种配置错误，是属于”可恢复的(recoverable/non-sticky)”错误
  * 一般来说我们可以将错误分为：
    * synchronous/asynchronous error
    * sticky/non-sticky error


#### 2.2.5 GPU的硬件信息获取

由于我们在模型优化的时候需要考虑硬件性能，让模型能够完全匹配硬件，所以需要获取GPU硬件信息。

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2f136c83-fc1f-4301-848b-064113a9b02b)


##### （1）新添加的内容

* utils.hpp
  * 把打印日志的方法定义成了一个宏，方便使用
  * VA_ARGS：编译器内部的定义的一个宏。
    * 表示的是变参。
    * 配合 vsnprintf 可以将 LOG 中的变参信息存入到 msg 的这个buffer 中。最终在一起打印出来
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/562ce996-ba15-4f2a-ba4c-ff66857cddca)




##### （2）为什么要注意硬件信息

* 1、我们需要知道我们的GPU的compute capability。nvcc的参数中有时会需要
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/eb144239-43ca-44f6-8607-0d503fb6af22)

* 2、我们需要知道我们在启动核函数的时候，配置信息的规定都有哪些
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4867ab89-9d51-4726-9a0d-52c580a58b40)

* 3、shared memory的使用对cuda程序的加速很重要。
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/47108d55-f15f-49ae-9c07-da021091ecd9)

  * 当我们在使用**shared memory**是需要知道它的大小上限是多少。
  * 这里需要注意的是，我们其实是可以调整shared memory和L1 cache在缓存中的空间。
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8d482b14-dd94-46b6-8280-60867258824c)

* 4、**warp scheduler**是cuda中对大量thread的一个调度器。我们需要知道一个warp是由多少个thread组成的。
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f8274c3a-1e39-4c29-b8c8-abd9a4c2c2f6)

* 5、当我们在进行性能调优的时候，内存的大小和内存的带宽是我们需要考虑的一个很重要的因素。结合 roofline model(后面的章节会讲，如何更好接近计算峰值)，我们需要寻找想要隐藏 memory 的数据传输所造成的overhead，需要多少的计算量和计算效率
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/5e14fa9f-31e9-44f1-9520-9bd80c62fa62)




### 2.3 Nsight systems and compute

这两个软件可以帮助我们分析kernel中的每个细节包括：shared memory的使用、读取写入数据的带宽情况、计算瓶颈等。同时能够帮助我们分析各个内容的调度。

#### 2.3.1 Nsight systems

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/7f7c4176-3ecf-42b3-a685-abb567bb4a3b)


#### 2.3.2 Nsight Compute

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e03445a4-7854-4e15-bbb4-915e545ad20e)


#### 2.3.3 主要应用

* 参考下面博客：
  * [CUDA编程 - Nsight system & Nsight compute 的安装和使用 - 学习记录_cuda nsight-CSDN博客](https://blog.csdn.net/weixin_40653140/article/details/136238420?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-136238420-blog-122532275.235^v43^pc_blog_bottom_relevance_base4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-136238420-blog-122532275.235^v43^pc_blog_bottom_relevance_base4&utm_relevant_index=2)



#### 2.3.4 配置流程

参考链接：

[在WSL2上运行nVIDIA Nsight_nsight wsl-CSDN博客](https://blog.csdn.net/cyr20040123/article/details/122532275?ops_request_misc=%7B%22request%5Fid%22%3A%22171280690916800213045012%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=171280690916800213045012&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-122532275-null-null.142^v100^pc_search_result_base5&utm_term=在WSL2上运行nVIDIA Nsight&spm=1018.2226.3001.4187)

##### （1）查询WSL2的SHH远程IP

* 在终端输入命令行``ifconfig``，查看 **eth0** 开头的 **inet** ：

  * 注意不是第一个docker0的IP，这个是无法连接的。

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c6f0a10f-89c7-423c-8c30-22969cdba6e0)




##### （2）配置SSH的config文件

* 安装openssh-server：

  * ```python
    sudo apt install openssh-server
    ```

* 打开config文件：

  * ```python
    sudo vim /etc/ssh/sshd_config
    ```

* 修改config内容：

  * 取消下面注释：

    * ```python
      Port 22
      AddressFamily any
      ListenAddress 0.0.0.0
      ListenAddress ::
          
      PasswordAuthentication yes
      ```

    * 修改内容：

      * Port 22
        AddressFamily any
        ListenAddress 0.0.0.0
        ListenAddress ::
        * 前面注释去掉
      * 把PasswordAuthentication 前面注释去掉，并且设置yes

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/31175506-7c98-45f2-8182-84ddff8dd3b9)




##### （3）重启SSH

* ```python
  sudo service ssh restart
  ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a8d010ea-b5c7-417b-b1de-d661433210a3)




##### （4）启动Nsight systems

参考教程：

[在WSL2上运行nVIDIA Nsight_nsight wsl-CSDN博客](https://blog.csdn.net/cyr20040123/article/details/122532275?ops_request_misc=%7B%22request%5Fid%22%3A%22171280690916800213045012%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=171280690916800213045012&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-122532275-null-null.142^v100^pc_search_result_base5&utm_term=在WSL2上运行nVIDIA Nsight&spm=1018.2226.3001.4187)

* 将自己的IP和用户名导入软件：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c63425ec-2a5b-4df6-b845-b5a219fc273e)

  * 报错：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/75d4b987-131b-452a-ac8e-fd96f4842604)

    * 取消红色感叹号框选项，具体解决办法已解决：
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/647d2d48-f749-43cf-92fd-d62464389d8d)


      * 参考下面连接：
        * [NVIDIA 开发工具解决方案 - |NVIDIA 开发人员](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
        * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/eac7d250-5865-4d21-8a7e-474765316137)



### 2.4 共享内存以及BANK CONFLICT

#### 2.4.1 如何使用共享内存

##### （1）新添加的东西

* 为了实现更加精准的kernel测速，这里采用了event进行标记。

  * 【timer.hpp】
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/05afccb9-d535-41b7-9f71-786c88734983)

  * 【timer.cpp】
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2f66aca6-b03c-47a2-8e04-337491806dca)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ff1e2681-e730-4479-ade5-07eca28b9b75)

  * event中文名字叫做"事件"，可以用来标记cuda的stream中某一个执行点。一般用在多个stream同步或者监听某个stream的执行。

* 使用静态共享变量【matmul_gpu_shared.cu】

  * ```c++
    /* 
        使用shared memory把计算一个tile所需要的数据分块存储到访问速度快的memory中
    */
    __global__ void MatmulSharedStaticKernel(float *M_device, float *N_device, float *P_device, int width){
        __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
        __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];
        /* 
            对于x和y, 根据blockID, tile大小和threadID进行索引
        */
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        float P_element = 0.0;
    
        int ty = threadIdx.y;
        int tx = threadIdx.x;
        /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了，这里有点绕，画图理解一下*/
        for (int m = 0; m < width / BLOCKSIZE; m ++) {
            M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
            N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty)* width + x];
            __syncthreads();
    
            for (int k = 0; k < BLOCKSIZE; k ++) {
                P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
            }
            __syncthreads();
        }
    
        P_device[y * width + x] = P_element;
    }
    ```

* 使用动态共享变量【matmul_gpu_shared.cu】

  * ```c++
    __global__ void MatmulSharedDynamicKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize){
        /* 
            声明动态共享变量的时候需要加extern，同时需要是一维的 
            注意这里有个坑, 不能够像这样定义： 
                __shared__ float M_deviceShared[];
                __shared__ float N_deviceShared[];
            因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
            所以如果想要像上面这样使用的话，需要用两个指针分别指向shared memory的不同位置才行
        */
    
        extern __shared__ float deviceShared[];
        int stride = blockSize * blockSize;
        /* 
            对于x和y, 根据blockID, tile大小和threadID进行索引
        */
        int x = blockIdx.x * blockSize + threadIdx.x;
        int y = blockIdx.y * blockSize + threadIdx.y;
    
        float P_element = 0.0;
    
        int ty = threadIdx.y;
        int tx = threadIdx.x;
        /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
        for (int m = 0; m < width / blockSize; m ++) {
            deviceShared[ty * blockSize + tx] = M_device[y * width + (m * blockSize + tx)];
            deviceShared[stride + (ty * blockSize + tx)] = N_device[(m * blockSize + ty)* width + x];
            __syncthreads();
    
            for (int k = 0; k < blockSize; k ++) {
                P_element += deviceShared[ty * blockSize + k] * deviceShared[stride + (k * blockSize + tx)];
            }
            __syncthreads();
        }
    
        if (y < width && x < width) {
            P_device[y * width + x] = P_element;
        }
    }
    ```

    

##### （2）矩阵乘法特点

* 从global memory中重复加载内存同一块空间：
  * c(0,0)、c(0,1)、c(0,2)、c(0,3)的乘法计算时候，都是从a(0,0)-->a(0,7)的A矩阵一行和B矩阵每列的各个元素反复相乘累加。
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d31ec367-c11b-42c4-b04f-7666beb68d3f)

  * c(0,0)、c(1,0)、c(2,0)、c(3,0)的乘法计算时候，都是从b(1,0)-->a(7,0)的B矩阵一列和C矩阵每行的各个元素反复相乘累加。
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1a6c4773-7f9f-41f2-a07b-decb8131c557)

* 因此，一直访问glogal memory的话有点慢（离计算单元越远的内存，访问时间就越慢）。如果数据可以在多次被复用的话，可以把他们放在可以快速访问的shared memory中
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9340e07b-c574-4ce3-8264-a065f962d077)


##### （3）Shared memory vs Global memory

* 区别：
  *  Shared memory: on-chip memory
  * Global memory:  off-chip memory (DRAM)
* 示例：
  * GPU：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/e0eae956-069c-4805-89f8-57abdc781130)

    *  L1/L2 cache和shared memory都是属于on-chip memory，memory  load/store的overhead会比较小，是可以高速访问的memory
    *  Global memory的延迟是最高的，我们一般在cudaMalloc时都是在global  memory上进行访问的
  * SM：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c4051ad7-26d5-4b1f-afe5-9848f77d7edb)

* on-chip memory
  * register：线程独享，访问最快大小最小
  * L1 cache：SM内共享
  * shared memory：SM内共享
  *  L2 cache：SM内共享
* off-chip memory
  * local memory：线程独享，寄存器不足的时候使用
  * Global memory：设备内所有线程共享
  * constant memory：只读内存，用与避免线程独数据冲突
  * texture memory：只读内存，tex可以实现硬件插值

##### （4）运行效果

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ef6aee4e-afd2-407d-9cc8-b9d306d50b28)


##### （5）代码细节

* 1、动态/静态共享变量切换：

  * 【main.cpp】

    * ```c++
          // /* GPU general implementation <<<256, 16>>>*/
          timer.start_gpu();
          MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
          timer.stop_gpu();
          std::sprintf(str, "matmul in gpu(with shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
          timer.duration_gpu(str);
          compareMat(h_matP, d_matP, size);
      
          /* GPU general implementation <<<256, 16>>>*/
          statMem = false;
          timer.start_gpu();
          MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
          timer.stop_gpu();
          std::sprintf(str, "matmul in gpu(with shared memory(dynamic))<<<%d, %d>>>", width / blockSize, blockSize);
          timer.duration_gpu(str);
          compareMat(h_matP, d_matP, size);
      ```

  * 【matmul_gpu_shared.cu】

    * ```C++
          /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
          dim3 dimBlock(blockSize, blockSize);
          dim3 dimGrid(width / blockSize, width / blockSize);
          if (staticMem) {
              MatmulSharedStaticKernel <<<dimGrid, dimBlock>>> (M_device, N_device, P_device, width);
          } else {
              MatmulSharedDynamicKernel <<<dimGrid, dimBlock, sMemSize, nullptr>>> (M_device, N_device, P_device, width, blockSize);
          }
      ```

* 2、静态共享内存

  * 【matmul_gpu_shared.cu】

    * ```c++
      /* 
          使用shared memory把计算一个tile所需要的数据分块存储到访问速度快的memory中
      */
      __global__ void MatmulSharedStaticKernel(float *M_device, float *N_device, float *P_device, int width){
          __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
          __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];
          /* 
              对于x和y, 根据blockID, tile大小和threadID进行索引
          */
          int x = blockIdx.x * blockDim.x + threadIdx.x;
          int y = blockIdx.y * blockDim.y + threadIdx.y;
      
          float P_element = 0.0;
      
          int ty = threadIdx.y;
          int tx = threadIdx.x;
          /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了，这里有点绕，画图理解一下*/
          // shared memory 写操作
          for (int m = 0; m < width / BLOCKSIZE; m ++) {
              M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
              N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty)* width + x];
              __syncthreads();    // 在一个Block里面对所有线程的同步，如果用到shared memory就要这样同步一下
      
              // shared memory 读操作，从shared memory中读数据
              for (int k = 0; k < BLOCKSIZE; k ++) {
                  P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
              }
              __syncthreads();
          }
      
          // global memory 写操作
          P_device[y * width + x] = P_element;
      }
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d01ee962-9bba-4d02-b5e0-e6a37a0e9185)


* 3、动态共享内存

  * 【matmul_gpu_shared.cu】

    * ```c++
      __global__ void MatmulSharedDynamicKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize){
          /* 
              声明动态共享变量的时候需要加extern，同时需要是一维的 
              注意这里有个坑, 不能够像这样定义： 
                  __shared__ float M_deviceShared[];
                  __shared__ float N_deviceShared[];
              因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
              所以如果想要像上面这样使用的话，需要用两个指针分别指向shared memory的不同位置才行
          */
      
          extern __shared__ float deviceShared[];
          int stride = blockSize * blockSize;
          /* 
              对于x和y, 根据blockID, tile大小和threadID进行索引
          */
          int x = blockIdx.x * blockSize + threadIdx.x;
          int y = blockIdx.y * blockSize + threadIdx.y;
      
          float P_element = 0.0;
      
          int ty = threadIdx.y;
          int tx = threadIdx.x;
          /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
          for (int m = 0; m < width / blockSize; m ++) {
              deviceShared[ty * blockSize + tx] = M_device[y * width + (m * blockSize + tx)];
              deviceShared[stride + (ty * blockSize + tx)] = N_device[(m * blockSize + ty)* width + x];
              __syncthreads();
      
              for (int k = 0; k < blockSize; k ++) {
                  P_element += deviceShared[ty * blockSize + k] * deviceShared[stride + (k * blockSize + tx)];
              }
              __syncthreads();
          }
      
          if (y < width && x < width) {
              P_device[y * width + x] = P_element;
          }
      }
      ```

    * 注意：声明动态共享变量的时候需要加extern，同时需要是一维的。

#### 2.4.2 理解BANK CONFLICT及其缓解策略

##### （1）新添加的东西

![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/140e4ab4-cc02-4c29-9621-dbd366780192)


##### （2）shared memory中存放数据的特殊方式

* 在cuda编程中，32个threads组成一个**warp**，一般程序在执行的时候是以**warp**为单位去执行，也就是说每32个threads一起执行同一个指令：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/502c2062-7e34-4b5d-9044-1dad9c3adb0b)

* 所以，为了能够高效的访存，shared memory中也对应了分成了32个**存储体**。我们称之为 “bank”，分别对应warp中32个线程：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ed106870-7c29-4fd4-ae0c-9305bc57df15)

* bank的宽度，代表的是一个bank所存储的数据的大小宽度。
  * 可以是4个字节(32 bit,  单精度浮点数float)
  * 也可以是8个字节(64bit，双精度浮点数)
* 每31个bank，就会进行一次stride。
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a64af915-685c-4f1d-afab-361837202820)

  *  比如说bank的宽度是4字节。我们在shared memory中申请了float A[256]大小的空间，那么：
    *  A[0], A[1], …, A[31]分别在bank0, bank1, …, bank31中
    *  A[32], A[33], …, A[63]也分在了bank0, bank1, …, bank31中
  * 所以A[0], A[32]是共享同一个bank的

##### （3）bank conflict

* 一个很理想的情况就是，32个thread，分别访问shared memory中的32个不同的bank：
  * **没有bank conflict**，一个memory周期完成所有的memory read/write (row major/行优先矩阵访问)
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/392ea48e-c9a8-4553-8ec1-7bc814676aec)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b09358ce-448d-4fab-8c7a-67a990de2d17)

  * 一个最不理想的情况就是，32个thread，访问shared memory中的同一个bank：
    * **bank conflict最大化**，需要32个memory周期才能完成所有的memory read/write (column major列优先矩阵访问)
      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3e16f76a-53f7-4816-bc34-4d1c0b8e45be)


      * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a972c8c8-0521-4db4-97cb-0f9c64dfff82)


##### （4）使用padding缓解bank conflict

* 在实际CUDA设计时，是32个bank一次stride，为了方便解释，这里使用了8个bank一次stride进行举例：
  * 当8个thread访问同一个bank[2]时，会产生bank conflict：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/49942c32-571b-4682-bc00-ec6ef2dc886a)

  * padding通过在这个8X8的shared memory矩阵最后多加一列，使得每8个bank进行一次stride，最终memory变成下图9X8的矩阵。
    * 这是每个thread在访问bank[2]时，都不会产生冲突了，因为每个bank[2]都已经不在同一个位置上了。
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f61625bb-b944-455e-ba93-61ff168969f4)


##### （5）运行效果

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f643d5af-664b-4578-affb-bb27c020604a)


##### （6）代码细节

* 【matmul_gpu_bank_conflict.cu】
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/91875d7a-5588-4d67-bd2c-6aece5f1bde9)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3cc938e0-57f1-429b-a799-0457a7ce925e)

* 【matmul_gpu_bank_conflict_pad.cu】
  * 静态：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b3505d6b-a27a-4ce6-862a-55330c273aed)

  * 动态：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d87cc911-80a6-4c10-8be5-9c36a39b1c9f)

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c93e3947-e9b3-401a-9a8d-8f565334fbed)




### 2.5 STREAM和EVENT

#### 2.5.1 新添加的内容

##### （1）文件夹列表

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/faee102d-882b-40e5-aad7-1059caadbbfd)

* 【stream.cu】
  * 为了能够平衡mempry以及kernel的执 行，这里在kernel内部设定了等待：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8d059bc2-531c-48cf-8d37-3011f4c0ead8)

  * 多流进行异步的计算流程：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/6e0ad1e6-f732-4828-aa0e-9b4ef50af066)

* 【gelu.cu】
  * 添加了一个GELU的CUDA实现，为后面搭建Plugin做铺垫
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/45fc2261-6a7b-4b8d-a89b-3b47b2e9ca15)


##### （2）运行结果

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4eec9da7-871f-4886-b02c-745e9bd25c51)


#### 2.5.2 Stream

##### （1）Stream简介

* “A sequence of operation that execute in issue-order in GPU”
* 同一个流的执行顺序和各个kernel以及mempry operation的启动的顺序是一致的。但是，只要资源没有被占用，不同流之间的执行是可以overlap的。
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/305d97e2-1d37-4c01-9d62-2d20d090c015)

  * PCIe是共享的，所以memcpy只能够在同一个时间执行一个
  * SM计算资源是有限的，所以如果计算资源占满了，多流和单流是差不多的

##### （2）默认流

* 当我们不指定核函数以及memcpy的流式，cuda会使用默认流(default stream)
* 完全串行，**H2D和D2H是在同一个steam中的不同队列**
  * 执行一个核函数：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c68fe86b-6344-48c4-b180-7268a1749208)

  * 执行两个不同的核函数：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/bdeea1f6-812e-4c30-805a-a16c4bc4bf98)


##### （3）显示的指定流进行操作

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/dd3df021-513f-4df1-b1bd-fc9df335726b)

* 注意：在显示的指定时候，在host端分配空间的时候，要用cudaMallocHost函数
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/b8eeb608-c9a3-4f55-9355-e408af552764)

* 当我们显式的指定流的时候，核函数是可能overlap：
  * 计算资源被沾满的时候：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f40999c7-c1ec-4b4f-87c2-780d2fc461a0)

  * 计算资源没有被占满的时候：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/adf74cf0-439d-4c55-bff6-83211d5a37a8)


##### （4）页锁定内存

* cudaMallocHost()函数使用的是页锁定内存：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d87bd3cb-f54d-4a29-975f-0f350c0baf8e)

* 内存分为：
  * Pageable memory：可分页内存
  * Pinned memory / page-locked memory：页锁定内存
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c1497906-8df3-409c-8e86-dc9231f83c0f)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/fa5beb07-c1b0-4526-a42c-22636d570b8f)

##### （5）Streams单流/多流对比

* 1、输入256X256大小的矩阵，Gridsize=4X4，Blocksize=4X4：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/ed530150-4c86-4cd4-aa9f-9ad64a4a6ef5)

* 2、输入256X256大小的矩阵，Gridsize=1X1，Blocksize=16X16：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/9395dd1a-b985-4cf0-adaa-69aab3308c96)

* 3、输入4096X4096大小的矩阵，Gridsize=4X4，Blocksize=16X16：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/1dcd7ef2-e3e4-42e8-b587-845f28e555d9)

* 4、输入104857X104857大小的矩阵，Gridsize=64X64，Blocksize=16X16：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/c97f9d09-d0b4-44a1-b0df-1e45ac9a1ecc)




##### （6）如何隐藏延迟(memory)

* 1、按正常的方式去调度流程（红色和绿色是两个不同的流）：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/71dbce7d-6d1d-4fc7-abd5-896381a678b1)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/eb8f9e8a-2e15-46db-b71b-16b5db779de8)

* 2、将D2H2放在一开始issue，可以减少D2H2 Queue的等待时间：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/8957617c-002e-477d-b025-ec00772d505d)


##### （7）如何隐藏延迟(kernel)

* 1、按正常的方式去调度流程（红色和绿色是两个不同的流）：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a5126d3f-0645-4673-a6e8-3b51bb514df5)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/f2d6773a-b2be-46ab-808e-9891e0b8afcc)

* 2、在KernelA1执行等待时launch KernelA2：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d9caf7af-c6a2-4d6e-9901-9ef2400e171c)

* 3、两个核函数一大一小时：
  * 正常方式：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/041aee78-7500-4f13-89ef-c38805697ad7)

  * 在KernelA1执行等待的时候启动A2：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2e3b7db0-c5bd-4a6e-a934-43d83d017e51)

  * A2先执行完发现可以先启动B2：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/0ad98a1d-b6e3-44af-98e2-e02b255d78ca)


##### （8）如何隐藏延迟(kernel + memory)

* ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/df44968d-9152-4e4e-bc96-279526b6c1f1)




#### 2.5.3 代码细节

##### （1）【Stream.cu】

* ```c++
  /* n stream，处理一次memcpy，以及n个kernel */
  void SleepMultiStream(
      float* src_host, float* tar_host,
      int width, int blockSize, 
      int count) 
  {
      int size = width * width * sizeof(float);
  
      float *src_device;
      float *tar_device;
  
      CUDA_CHECK(cudaMalloc((void**)&src_device, size));
      CUDA_CHECK(cudaMalloc((void**)&tar_device, size));
  
  
      /* 先把所需要的stream创建出来 */
      cudaStream_t stream[count]; // 创建一个stream变量
      for (int i = 0; i < count ; i++) {
          CUDA_CHECK(cudaStreamCreate(&stream[i])); //利用cudaStreamCreate创建stream
      }
  
      for (int i = 0; i < count ; i++) {
          for (int j = 0; j < 1; j ++) 
              // 异步的memory copy，指定stream
              CUDA_CHECK(cudaMemcpyAsync(src_device, src_host, size, cudaMemcpyHostToDevice, stream[i]));
          dim3 dimBlock(blockSize, blockSize);
          dim3 dimGrid(width / blockSize, width / blockSize);
  
          /* 这里面我们把参数写全了 <<<dimGrid, dimBlock, sMemSize, stream>>> */
          SleepKernel <<<dimGrid, dimBlock, 0, stream[i]>>> (MAX_ITER);
          CUDA_CHECK(cudaMemcpyAsync(src_host, src_device, size, cudaMemcpyDeviceToHost, stream[i]));
      }
  
  
      CUDA_CHECK(cudaDeviceSynchronize());
  
  
      cudaFree(tar_device);
      cudaFree(src_device);
  
      // 释放stream
      for (int i = 0; i < count ; i++) {
          // 使用完了以后不要忘记释放
          cudaStreamDestroy(stream[i]);
      }
  
  }
  ```



### 2.6 双线性插值与仿射变换

由于我们在终端部署的时候，输入的图片尺寸会不同，因此统一图片尺寸是一个非常必要的工作。这时需要对图片resize，如果使用openCV的resize的话，会非常的慢。

#### 2.6.1 双线性插值与仿射变换简介

##### （1）不同的双线性插值

* 原图（图1）、普通的双线性插值scale转换成高宽比一样的图片（图2）、scale保持不变的resized（图3）、scale保持不变并居中的resized（图4）
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/211706a0-36ad-46ec-bd4d-3bcc92754be1)

* 执行效果：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/69727d50-f556-4a38-9c23-dee948c47be0)




##### （2）普通的双线性插值

* 是一种对图像进行缩放/放大的一种计算方法，scale转换成高宽比一样的图片：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/d121569d-4ee4-4196-afe5-350fcb5080fc)

* 由于像素点在scale以后，精度会发生变化，因此在256X256分辨率下找到的点（浮点数强制转换为int），需要计算RGB值，openCV提供了最临近法：*Nearest neighbor*方法。将resize之后的坐标点返回到原图片中，找到离该浮点坐标最近的点，如下图所示：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a1485504-6430-4a82-8863-5b18b6fe27a0)

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a9c23a11-cf97-46b0-b833-546a2ec776d3)


##### （3）双线性插值（宽高比未复原）

* 不仅要考虑还原后最近的坐标点，还要考虑它附近所有四个点坐标，以及他们与中间点之间的面积：
  * resize后的target坐标：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/990da94e-984f-4e57-8f00-0c4ab32fd507)

  * 还原后的坐标以及周围的四个点坐标：
    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/2fe5366b-a674-4d49-9d5a-c40c39336439)

* 双线性插值计算最终RGB值：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4ea3d75d-3496-47b4-a994-f6225ed8d093)


  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/a9c14090-9540-4852-bbc9-1e61a107e697)


##### （4）双线性插值（复原高宽比）

* 如果我们想要缩放后依然保持比例的话，我们对宽和高的缩放比保持一致就可以了：

  * 计算缩放比公式：

    * ```c++
      //scaled resize
      float scaled_h = (float)srcH / tarH;
      float scaled_w = (float)srcW / tarW;
      float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);
      ```

    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/4025fbbe-b7d2-4a4e-99d1-b989bd85c8e6)


    * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/28fd8cc9-95cc-434b-b2ae-67f7ba74292e)


##### （5）双线性插值（复原高宽比居中）

* 一般来说，在进行缩放以后我们希望让图像居中，所以需要让图像的中心坐标shift一定的像素：

  * ```c++
    // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
    y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
    x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);
    ```

  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/60a95fc7-a952-4bbf-add5-ceaebab23be1)




##### （6）BGR2RGB

* opencv读取完图片以后，默认的格式是BGR的。如果要将读取的图片传给DNN进行推理的话，我们需要将channel 的方向改变一下。在CUDA中可以这么更改：
  * ![image](https://github.com/CoderSuHang/TensorRT-Learning-Note/assets/104765251/3c3302cb-ca84-4ebe-8219-9f53668b7bc6)

  * 其实，如果需要的话，我们也可以在BGR2RGB的同时，实现BHWC2BCHW的转变。类似于PyTorch中的transpose

#### 2.6.2 新的内容

##### （1）2.10-bilinear-interpolation

* 普通的双线性插值
  * 内部核函数主要做的是uint8的核函数计算
  * 更偏向实际应用
* ![image-20240416165501869](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240416165501869.png)

##### （2）2.11-bilinear-interpolation-template

* 模板函数
  * 内部核函数可以根据用户的使用情况做uint38、float32等的核函数计算
  * 更偏向实际应用



#### 2.6.3 代码详情

##### （1）2.10-bilinear-interpolation

* 执行结果：

  * ![image-20240418103736444](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240418103736444.png)
  * CPU与GPU之间有20倍左右的加速效果

* OpenCV导入图片：

  * ```c++
    #include <stdio.h>
    #include <cuda_runtime.h>
    #include <iostream>
    
    #include "utils.hpp"
    #include "timer.hpp"
    #include "preprocess.hpp"
    
    using namespace std;
    
    int main(){
        Timer timer;
    
        string file_path     = "data/deer.png";
        string output_prefix = "results/";
        string output_path   = "";
    
        cv::Mat input = cv::imread(file_path);
        int tar_h = 500;
        int tar_w = 250;
        int tactis;
    
        cv::Mat resizedInput_cpu;
        cv::Mat resizedInput_gpu;
        
        /* 
         * bilinear interpolation resize的CPU/GPU速度比较
         * tatics 列表
         * 0: 最近邻差值缩放 + 全图填充
         * 1: 双线性差值缩放 + 全图填充
         * 2: 双线性差值缩放 + 填充(letter box)
         * 3: 双线性差值缩放 + 填充(letter box) + 平移居中
         * */
        
        resizedInput_cpu = preprocess_cpu(input, tar_h, tar_w, timer, tactis);
        output_path = output_prefix + getPrefix(file_path) + "_resized_bilinear_cpu.png";
        cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
        cv::imwrite(output_path, resizedInput_cpu);
    
        tactis = 0;
        resizedInput_gpu = preprocess_gpu(input, tar_h, tar_w, timer, tactis);
        output_path = output_prefix + getPrefix(file_path) + "_resized_nearest_gpu.png";
        cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
        cv::imwrite(output_path, resizedInput_gpu);
    
        tactis = 1;
        resizedInput_gpu = preprocess_gpu(input, tar_h, tar_w, timer, tactis);
        output_path = output_prefix + getPrefix(file_path) + "_resized_bilinear_gpu.png";
        cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
        cv::imwrite(output_path, resizedInput_gpu);
    
        tactis = 2;
        resizedInput_gpu = preprocess_gpu(input, tar_h, tar_w, timer, tactis);
        output_path = output_prefix + getPrefix(file_path) + "_resized_bilinear_letterbox_gpu.png";
        cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
        cv::imwrite(output_path, resizedInput_gpu);
    
        tactis = 3;
        resizedInput_gpu = preprocess_gpu(input, tar_h, tar_w, timer, tactis);
        output_path = output_prefix + getPrefix(file_path) + "_resized_bilinear_letterbox_center_gpu.png";
        cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
        cv::imwrite(output_path, resizedInput_gpu);
        return 0;
    }
    ```

* CPU实现：

  * ```c++
    // 根据比例进行缩放 (CPU版本) bilinear resize的opencv实现
    cv::Mat preprocess_cpu(cv::Mat &src, const int &tar_h, const int &tar_w, Timer timer, int tactis) {
        cv::Mat tar;
        
        // 先找到resizeH和W的大小
        int height  = src.rows;
        int width   = src.cols;
        float dim   = std::max(height, width);
        int resizeH = ((height / dim) * tar_h);
        int resizeW = ((width / dim) * tar_w);
    
        int xOffSet = (tar_w - resizeW) / 2;
        int yOffSet = (tar_h - resizeH) / 2;
    
        resizeW    = tar_w;
        resizeH    = tar_h;
    
        /*测速*/
        timer.start_cpu();
    
        /*BGR2RGB*/
        cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
    
        /*Resize*/
        cv::resize(src, tar, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);
    
        timer.stop_cpu();
        timer.duration_cpu<Timer::ms>("Resize(bilinear) in cpu takes:");
    
        return tar;
    }
    ```

* GPU实现：

  * ```c++
    #include "preprocess.hpp"
    #include "opencv2/opencv.hpp"
    #include "utils.hpp"
    #include "timer.hpp"
    
    // 根据比例进行缩放 (CPU版本) bilinear resize的opencv实现
    cv::Mat preprocess_cpu(cv::Mat &src, const int &tar_h, const int &tar_w, Timer timer, int tactis) {
        cv::Mat tar;
        
        // 先找到resizeH和W的大小
        int height  = src.rows;
        int width   = src.cols;
        float dim   = std::max(height, width);
        int resizeH = ((height / dim) * tar_h);
        int resizeW = ((width / dim) * tar_w);
    
        int xOffSet = (tar_w - resizeW) / 2;
        int yOffSet = (tar_h - resizeH) / 2;
    
        resizeW    = tar_w;
        resizeH    = tar_h;
    
        /*测速*/
        timer.start_cpu();
    
        /*BGR2RGB*/
        cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
    
        /*Resize*/
        cv::resize(src, tar, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_LINEAR);
    
        timer.stop_cpu();
        timer.duration_cpu<Timer::ms>("Resize(bilinear) in cpu takes:");
    
        return tar;
    }
    
    // 根据比例进行缩放 (GPU版本) bilinear resize的cuda实现（接口部分）
    cv::Mat preprocess_gpu(
        cv::Mat &h_src, const int& tar_h, const int& tar_w, Timer timer, int tactis) 
    {
        uint8_t* d_tar = nullptr;
        uint8_t* d_src = nullptr;
    
        // 分配空间大小CV_8UC3：UC=unsigned char，3=chanel，8=uint8
        cv::Mat h_tar(cv::Size(tar_w, tar_h), CV_8UC3);
    
        int height   = h_src.rows;
        int width    = h_src.cols;
        int chan     = 3;
    
        int src_size  = height * width * chan;
        int tar_size  = tar_h * tar_w * chan;
    
        // 分配device上的src和tar的内存
        CUDA_CHECK(cudaMalloc(&d_src, src_size));
        CUDA_CHECK(cudaMalloc(&d_tar, tar_size));
    
        // 将数据拷贝到device上
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyHostToDevice));
    
        timer.start_gpu();
    
        // device上处理resize, BGR2RGB的核函数
        resize_bilinear_gpu(d_tar, d_src, tar_w, tar_h, width, height, tactis);
    
        // host和device进行同步处理
        CUDA_CHECK(cudaDeviceSynchronize());
    
        timer.stop_gpu();
        switch (tactis) {
            case 0: timer.duration_gpu("Resize(nearest) in gpu takes:"); break;
            case 1: timer.duration_gpu("Resize(bilinear) in gpu takes:"); break;
            case 2: timer.duration_gpu("Resize(bilinear-letterbox) in gpu takes:"); break;
            case 3: timer.duration_gpu("Resize(bilinear-letterbox-center) in gpu takes:"); break;
            default: break;
        }
    
        // 将结果返回给host上
        CUDA_CHECK(cudaMemcpy(h_tar.data, d_tar, tar_size, cudaMemcpyDeviceToHost));
    
    
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_tar));
    
        return h_tar;
    }
    ```

  * ![image-20240418110619543](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240418110619543.png)

  * ![image-20240418110730724](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240418110730724.png)

* 【preprocess.cu】双线性插值算法实现

  * ```c++
    #include "cuda_runtime_api.h"
    #include "stdio.h"
    #include <iostream>
    
    #include "utils.hpp"
    
    
    __global__ void resize_nearest_BGR2RGB_kernel(
        uint8_t* tar, uint8_t* src, 
        int tarW, int tarH, 
        int srcW, int srcH,
        float scaled_w, float scaled_h) 
    {
        // nearest neighbour -- resized之后的图tar上的坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        // nearest neighbour -- 计算最近坐标
        int src_y = round((float)y * scaled_h);
        int src_x = round((float)x * scaled_w);
    
        if (src_x < 0 || src_y < 0 || src_x > srcW || src_y > srcH) {
            // nearest neighbour -- 对于越界的部分，不进行计算
        } else {
            // nearest neighbour -- 计算tar中对应坐标的索引
            // 3是chanel的个数，数据都是1维的，每个坐标乘3就是索引
            int tarIdx = (y * tarW  + x) * 3;
    
            // nearest neighbour -- 计算src中最近邻坐标的索引
            int srcIdx = (src_y * srcW + src_x) * 3;
    
            // nearest neighbour -- 实现nearest beighbour的resize + BGR2RGB
            tar[tarIdx + 0] = src[srcIdx + 2];
            tar[tarIdx + 1] = src[srcIdx + 1];
            tar[tarIdx + 2] = src[srcIdx + 0];
        }
    }
    
    __global__ void resize_bilinear_BGR2RGB_kernel(
        uint8_t* tar, uint8_t* src, 
        int tarW, int tarH, 
        int srcW, int srcH, 
        float scaled_w, float scaled_h) 
    {
    
        // bilinear interpolation -- resized之后的图tar上的坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
        // src_y1，src_x1就是ppt里面的（200，100）这个点（左上角）坐标
        int src_y1 = floor((y + 0.5) * scaled_h - 0.5); // floor取整
        int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
        // src_y2，src_x2就是ppt里面的（201，101）这个点（右下角）坐标
        int src_y2 = src_y1 + 1;
        int src_x2 = src_x1 + 1;
    
        if (src_y1 < 0 || src_x1 < 0 || src_y1 > srcH || src_x1 > srcW) {
            // bilinear interpolation -- 对于越界的坐标不进行计算
        } else {
            // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
            // th，tw就是PPT中resized点还原后的坐标转换成在0~1之间的值
            float th   = ((y + 0.5) * scaled_h - 0.5) - src_y1;
            float tw   = ((x + 0.5) * scaled_w - 0.5) - src_x1;
    
            // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
            float a1_1 = (1.0 - tw) * (1.0 - th);  //右下
            float a1_2 = tw * (1.0 - th);          //左下
            float a2_1 = (1.0 - tw) * th;          //右上
            float a2_2 = tw * th;                  //左上
    
            // bilinear interpolation -- 计算4个坐标所对应的索引
            int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  //左上
            int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  //右上
            int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  //左下
            int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  //右下
    
            // bilinear interpolation -- 计算resized之后的图的索引
            int tarIdx    = (y * tarW  + x) * 3;
    
            // bilinear interpolation -- 实现bilinear interpolation的resize + BGR2RGB
            tar[tarIdx + 0] = round(
                              a1_1 * src[srcIdx1_1 + 2] + 
                              a1_2 * src[srcIdx1_2 + 2] +
                              a2_1 * src[srcIdx2_1 + 2] +
                              a2_2 * src[srcIdx2_2 + 2]);
    
            tar[tarIdx + 1] = round(
                              a1_1 * src[srcIdx1_1 + 1] + 
                              a1_2 * src[srcIdx1_2 + 1] +
                              a2_1 * src[srcIdx2_1 + 1] +
                              a2_2 * src[srcIdx2_2 + 1]);
    
            tar[tarIdx + 2] = round(
                              a1_1 * src[srcIdx1_1 + 0] + 
                              a1_2 * src[srcIdx1_2 + 0] +
                              a2_1 * src[srcIdx2_1 + 0] +
                              a2_2 * src[srcIdx2_2 + 0]);
        }
    }
    
    __global__ void resize_bilinear_BGR2RGB_shift_kernel(
        uint8_t* tar, uint8_t* src, 
        int tarW, int tarH, 
        int srcW, int srcH, 
        float scaled_w, float scaled_h) 
    {
    
        // resized之后的图tar上的坐标
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
        int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
        int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
        int src_y2 = src_y1 + 1;
        int src_x2 = src_x1 + 1;
    
        if (src_y1 < 0 || src_x1 < 0 || src_y1 > srcH || src_x1 > srcW) {
            // bilinear interpolation -- 对于越界的坐标不进行计算
        } else {
            // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
            float th   = ((y + 0.5) * scaled_h - 0.5) - src_y1;
            float tw   = ((x + 0.5) * scaled_w - 0.5) - src_x1;
    
            // bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
            float a1_1 = (1.0 - tw) * (1.0 - th);  //右下
            float a1_2 = tw * (1.0 - th);          //左下
            float a2_1 = (1.0 - tw) * th;          //右上
            float a2_2 = tw * th;                  //左上
    
            // bilinear interpolation -- 计算4个坐标所对应的索引
            int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  //左上
            int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  //右上
            int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  //左下
            int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  //右下
    
            // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
            y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
            x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);
    
            // bilinear interpolation -- 计算resized之后的图的索引
            int tarIdx    = (y * tarW  + x) * 3;
    
            // bilinear interpolation -- 实现bilinear interpolation + BGR2RGB
            tar[tarIdx + 0] = round(
                              a1_1 * src[srcIdx1_1 + 2] + 
                              a1_2 * src[srcIdx1_2 + 2] +
                              a2_1 * src[srcIdx2_1 + 2] +
                              a2_2 * src[srcIdx2_2 + 2]);
    
            tar[tarIdx + 1] = round(
                              a1_1 * src[srcIdx1_1 + 1] + 
                              a1_2 * src[srcIdx1_2 + 1] +
                              a2_1 * src[srcIdx2_1 + 1] +
                              a2_2 * src[srcIdx2_2 + 1]);
    
            tar[tarIdx + 2] = round(
                              a1_1 * src[srcIdx1_1 + 0] + 
                              a1_2 * src[srcIdx1_2 + 0] +
                              a2_1 * src[srcIdx2_1 + 0] +
                              a2_2 * src[srcIdx2_2 + 0]);
        }
    }
    
    /*
        这里面的所有函数都实现了kernel fusion。这样可以减少kernel launch所产生的overhead
        如果使用了shared memory的话，就可以减少分配shared memory所产生的overhead以及内部线程同步的overhead。(这个案例没有使用shared memory)
        CUDA编程中有一些cuda runtime api是implicit synchronize(隐式同步)的，比如cudaMalloc, cudaMallocHost，以及shared memory的分配。
        高效的CUDA编程需要意识这些implicit synchronize以及其他会产生overhead的地方。比如使用内存复用的方法，让cuda分配完一次memory就一直使用它
    
        这里建议大家把我写的每一个kernel都拆开成不同的kernel来分别计算
        e.g. resize kernel + BGR2RGB kernel + shift kernel 
        之后用nsight去比较融合与不融合的差别在哪里。去体会一下fusion的好处
    */
    
    void resize_bilinear_gpu(
        uint8_t* d_tar, uint8_t* d_src, 
        int tarW, int tarH, 
        int srcW, int srcH, 
        int tactis) 
    {
        dim3 dimBlock(16, 16, 1);
        dim3 dimGrid(tarW / 16 + 1, tarH / 16 + 1, 1);
        
        //scaled resize
        float scaled_h = (float)srcH / tarH;
        float scaled_w = (float)srcW / tarW;
        float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);
    
        if (tactis > 1) {
            scaled_h = scale;
            scaled_w = scale;
        }
        
        switch (tactis) {
        case 0:
            resize_nearest_BGR2RGB_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
            break;
        case 1:
            resize_bilinear_BGR2RGB_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
            break;
        case 2:
            resize_bilinear_BGR2RGB_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
            break;
        case 3:
            resize_bilinear_BGR2RGB_shift_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
            break;
        default:
            break;
        }
    }
    ```



* 不同输出大小下的加速效果：
  * 750 x 750：
    * ![image-20240418103736444](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240418103736444.png)
  * 1000 x 750：
    * ![image-20240418114033406](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240418114033406.png)
  * 500 x 250：
    * ![image-20240418114124814](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240418114124814.png)

##### （2）2.11-bilinear-interpolation-template

* 运行效果：

  * ![image-20240418114521081](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240418114521081.png)

* 【main.cpp】设置计算输入精度

  * ![image-20240418114315521](C:\Users\10482\AppData\Roaming\Typora\typora-user-images\image-20240418114315521.png)

* 【preprocess.cpp】

  * ```c++
    // 根据比例进行缩放 (GPU版本)
    template <typename T>
    cv::Mat preprocess_gpu(
        cv::Mat &h_src, const int& tar_h, const int& tar_w, Timer timer) 
    {
        T*       d_tar = nullptr;
        uint8_t* d_src = nullptr;
    
        // 这里针对传入进来的type进行判断
        // 如果做fp32分配空间会不同，因此要根据不同的type分配不同的资源大小
        // 如果输入fp32计算，但是分配uint8空间大小，那么输出的图就是一片黑
        int type = (std::is_same<T, uint8_t>::value) ? CV_8UC3 : CV_32FC3;
        cv::Mat h_tar(cv::Size(tar_w, tar_h), type);
    
        int height   = h_src.rows;
        int width    = h_src.cols;
        int chan     = 3;
    
        int src_size  = height * width * chan * sizeof(uint8_t);
        int tar_size  = tar_h * tar_w * chan * sizeof(T);
    
        // 分配device上的src和tar的内存
        CUDA_CHECK(cudaMalloc(&d_src, src_size));
        CUDA_CHECK(cudaMalloc(&d_tar, tar_size));
    
        // 将数据拷贝到device上
        CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyHostToDevice));
    
        timer.start_gpu();
    
        // device上处理resize, BGR2RGB的核函数
        resize_bilinear_gpu(d_tar, d_src, tar_w, tar_h, width, height);
    
        // host和device进行同步处理
        CUDA_CHECK(cudaDeviceSynchronize());
    
        timer.stop_gpu();
        if (type == CV_8UC3) {
            timer.duration_gpu("Resize(bilinear-letterbox-center, uint8) in gpu takes:");
        } else {
            timer.duration_gpu("Resize(bilinear-letterbox-center, float32) in gpu takes:");
        }
    
        // 将结果返回给host上
        CUDA_CHECK(cudaMemcpy(h_tar.data, d_tar, tar_size, cudaMemcpyDeviceToHost));
    
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_tar));
    
        return h_tar;
    }
    
    
    template cv::Mat preprocess_gpu<u_int8_t>(cv::Mat &h_src, const int& tar_h, const int& tar_w, Timer timer);
    template cv::Mat preprocess_gpu<float>(cv::Mat &h_src, const int& tar_h, const int& tar_w, Timer timer) ;
    ```

