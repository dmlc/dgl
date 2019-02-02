@ECHO OFF
SETLOCAL EnableDelayedExpansion

DEL /S /Q build
DEL /S /Q _download
MD build

PUSHD build
CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" .. -G "Visual Studio 15 2017 Win64" || EXIT /B 1
msbuild dgl.sln || EXIT /B 1
COPY Release\dgl.dll .
POPD

PUSHD python
DEL /S /Q build *.egg-info dist
pip install -e . --force-reinstall --user || EXIT /B 1
POPD

ENDLOCAL
EXIT /B
