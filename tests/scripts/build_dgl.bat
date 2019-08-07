@ECHO OFF
SETLOCAL EnableDelayedExpansion

CALL mkvirtualenv --system-site-packages %BUILD_TAG%
DEL /S /Q build
DEL /S /Q _download
MD build

PUSHD build
CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DUSE_OPENMP=ON -Dgtest_force_shared_crt=ON -DDMLC_FORCE_SHARED_CRT=ON -DBUILD_CPP_TEST=1 -DCMAKE_CONFIGURATION_TYPES="Release" .. -G "Visual Studio 15 2017 Win64" || EXIT /B 1
msbuild dgl.sln || EXIT /B 1
COPY Release\dgl.dll .
COPY Release\runUnitTests.exe .
POPD

PUSHD python
DEL /S /Q build *.egg-info dist
pip install -e . || EXIT /B 1
POPD

ENDLOCAL
EXIT /B
