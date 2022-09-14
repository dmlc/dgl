REM Needs vcvars64.bat to be called
SET CUDADIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA
SET OLDPATH=%PATH%
SET CUDA_PATH=%CUDADIR%\v%CUDA_VER%
SET PATH=%CUDA_PATH%;%OLDPATH%
SET CUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH%
git submodule init
git submodule update
MD build
CD build
MD tensoradapter
MD tensoradapter\pytorch

IF x%CACHEDIR%x == xx (
	ECHO No prebuilt binary directory specified, building with default options...
	cmake -DUSE_CUDA=%USE_CUDA% -DUSE_OPENMP=ON -DCUDA_ARCH_NAME=All -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" .. -G "Visual Studio 15 2017 Win64" || EXIT /B 1
	msbuild dgl.sln || EXIT /B 1
	COPY Release\dgl.dll . || EXIT /B 1
) ELSE (
	COPY %CACHEDIR%\dgl%CUDA_VER%.dll .\dgl.dll || EXIT /B 1
	COPY %CACHEDIR%\tensoradapter-pytorch-%CUDA_VER%\*.dll tensoradapter\pytorch || EXIT /B 1
)
cd ..\python
"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt || EXIT /B 1
EXIT /B
