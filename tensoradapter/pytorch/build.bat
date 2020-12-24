REM Helper script to build tensor adapter libraries for PyTorch
@ECHO OFF
SETLOCAL EnableDelayedExpansion

DEL /S /Q build
MD build
PUSHD build

IF x%1x == xx GOTO single

FOR %%X IN (%*) DO (
	DEL /S /Q *
	%CMAKE_COMMAND% -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_TOOLKIT_ROOT_DIR% -DPYTHON_INTERP=%%X .. -G "Visual Studio 16 2019" || EXIT /B 1
	msbuild tensoradapter_pytorch.sln /m /nr:false || EXIT /B 1
	COPY /Y Release\*.dll %BINDIR%\tensoradapter\pytorch || EXIT /B 1
)

GOTO end

:single

DEL /S /Q *
%CMAKE_COMMAND% -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_TOOLKIT_ROOT_DIR% .. -G "Visual Studio 16 2019" || EXIT /B 1
msbuild tensoradapter_pytorch.sln /m /nr:false || EXIT /B 1
COPY /Y Release\*.dll %BINDIR%\tensoradapter\pytorch || EXIT /B 1

:end
MD %BINDIR%\tensoradapter\pytorch
POPD

ENDLOCAL
