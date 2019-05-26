@ECHO OFF
SETLOCAL EnableDelayedExpansion

DEL /S /Q build
DEL /S /Q _download
MD build

PUSHD build
CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DBUILD_CPP_TEST=1 -DCMAKE_CONFIGURATION_TYPES="Release" .. -G "Visual Studio 15 2017 Win64" || EXIT /B 1
msbuild dgl.sln || EXIT /B 1
./runUnitTests
COPY Release\dgl.dll .
POPD
