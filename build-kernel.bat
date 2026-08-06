@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
where cl
nvcc -allow-unsupported-compiler -ptx -arch=compute_61 -O3 -use_fast_math -Wno-deprecated-gpu-targets kernel.cu -o kernel_61.ptx
nvcc -allow-unsupported-compiler -ptx -arch=compute_120 -O3 -use_fast_math kernel.cu -o kernel_120.ptx
echo NVCC_EXIT=%ERRORLEVEL%
