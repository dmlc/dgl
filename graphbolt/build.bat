REM Helper script to build Graphbolt libraries for PyTorch
@ECHO OFF
SETLOCAL EnableDelayedExpansion

MD "%BINDIR%\graphbolt"
DEL /S /Q build
MD build
PUSHD build

IF x%1x == xx GOTO single

FOR %%X IN (%*) DO (
  DEL /S /Q *
  "%CMAKE_COMMAND%" -DGPU_CACHE_BUILD_DIR=%BINDIR% -DCMAKE_CONFIGURATION_TYPES=Release -DPYTHON_INTERP=%%X -DTORCH_CUDA_ARCH_LIST=Volta .. -G "Visual Studio 16 2019" || EXIT /B 1
  msbuild graphbolt.sln /m /nr:false || EXIT /B 1
  COPY /Y Release\*.dll "%BINDIR%\graphbolt" || EXIT /B 1
)

GOTO end

:single

DEL /S /Q *
"%CMAKE_COMMAND%" -DGPU_CACHE_BUILD_DIR=%BINDIR% -DCMAKE_CONFIGURATION_TYPES=Release -DTORCH_CUDA_ARCH_LIST=Volta .. -G "Visual Studio 16 2019" || EXIT /B 1
msbuild graphbolt.sln /m /nr:false || EXIT /B 1
COPY /Y Release\*.dll "%BINDIR%\graphbolt" || EXIT /B 1

:end
POPD

ENDLOCAL
