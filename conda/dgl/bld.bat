REM Needs vcvars64.bat to be called
git submodule init
git submodule update
md build
cd build
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" .. -G "Visual Studio 15 2017 Win64" || EXIT /B 1
msbuild dgl.sln || EXIT /B 1
COPY Release\dgl.dll .
cd ..\python
"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt || EXIT /B 1
EXIT /B
