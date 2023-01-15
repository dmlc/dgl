REM Helper script to build DGL sparse libraries for PyTorch
@ECHO OFF
SETLOCAL EnableDelayedExpansion

MD "%BINDIR%\dgl_sparse"
DEL /S /Q build
MD build
PUSHD build

IF x%1x == xx GOTO single
COPY %BINDIR%\third_party\dmlc-core\Release\dmlc.lib %BINDIR%
COPY %BINDIR%\Release\dgl.lib %BINDIR%

FOR %%X IN (%*) DO (
	DEL /S /Q *
	"%CMAKE_COMMAND%" -DDGL_BUILD_DIR=%BINDIR% -DCMAKE_CONFIGURATION_TYPES=Release -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DTORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST% -DDGL_INCLUDE_DIRS=%INCLUDEDIR: =;% -DUSE_CUDA=%USE_CUDA% -DPYTHON_INTERP=%%X .. -G "Visual Studio 16 2019" || EXIT /B 1
	msbuild dgl_sparse.sln /m /nr:false || EXIT /B 1
	COPY /Y Release\*.dll "%BINDIR%\dgl_sparse" || EXIT /B 1
)

GOTO end

:single

DEL /S /Q *
"%CMAKE_COMMAND%" -DDGL_BUILD_DIR=%BINDIR% -DCMAKE_CONFIGURATION_TYPES=Release -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DTORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST% -DUSE_CUDA=%USE_CUDA% -DDGL_INCLUDE_DIRS=%INCLUDEDIR: =;% .. -G "Visual Studio 16 2019" || EXIT /B 1
msbuild dgl_sparse.sln /m /nr:false || EXIT /B 1
COPY /Y Release\*.dll "%BINDIR%\dgl_sparse" || EXIT /B 1

:end
POPD

ENDLOCAL
