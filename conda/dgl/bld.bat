REM Needs vcvars64.bat to be called
git submodule init
git submodule update --recursive
md build
cd build
cmake -DUSE_CUDA=%USE_CUDA% -DUSE_OPENMP=ON -DCUDA_ARCH_NAME=All -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" -DDMLC_FORCE_SHARED_CRT=ON .. -G "Visual Studio 15 2017 Win64" || EXIT /B 1
msbuild dgl.sln || EXIT /B 1
COPY Release\dgl.dll .
cd ..\python
"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt || EXIT /B 1
EXIT /B
