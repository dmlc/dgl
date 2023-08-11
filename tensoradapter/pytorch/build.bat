REM Helper script to build tensor adapter libraries for PyTorch
@ECHO OFF
SETLOCAL EnableDelayedExpansion

MD "%BINDIR%\tensoradapter\pytorch"
DEL /S /Q build
MD build
PUSHD build

IF x%1x == xx GOTO single

CMAKE_FLAGS="-DCMAKE_C_COMPILER=$CMAKE_C_COMPILER -DCMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER"

FOR %%X IN (%*) DO (
	DEL /S /Q *
	"%CMAKE_COMMAND%" -DCMAKE_CONFIGURATION_TYPES=Release -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DTORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST% -DUSE_CUDA=%USE_CUDA% -DCMAKE_C_COMPILER=%CMAKE_C_COMPILER% -DCMAKE_CXX_COMPILER=%CMAKE_CXX_COMPILER% -DPYTHON_INTERP=%%X .. -G "Visual Studio 16 2019" || EXIT /B 1
	msbuild tensoradapter_pytorch.sln /m /nr:false || EXIT /B 1
	COPY /Y Release\*.dll "%BINDIR%\tensoradapter\pytorch" || EXIT /B 1
)

GOTO end

:single

DEL /S /Q *
"%CMAKE_COMMAND%" -DCMAKE_CONFIGURATION_TYPES=Release -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DTORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST% -DUSE_CUDA=%USE_CUDA% -DCMAKE_C_COMPILER=%CMAKE_C_COMPILER% -DCMAKE_CXX_COMPILER=%CMAKE_CXX_COMPILER% .. -G "Visual Studio 16 2019" || EXIT /B 1
msbuild tensoradapter_pytorch.sln /m /nr:false || EXIT /B 1
COPY /Y Release\*.dll "%BINDIR%\tensoradapter\pytorch" || EXIT /B 1

:end
POPD

ENDLOCAL
