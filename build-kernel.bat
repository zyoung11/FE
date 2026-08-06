@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
where cl
nvcc -allow-unsupported-compiler -ptx -arch=compute_80 -O3 -use_fast_math kernel.cu -o kernel.ptx
echo NVCC_EXIT=%ERRORLEVEL%
