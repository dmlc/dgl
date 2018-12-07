git submodule init
git submodule update
md build
cd build
cmake -DCMAKE_CXX_FLAGS="-DDMLC_LOG_STACK_TRACE=0 -DDGL_EXPORTS" -DCMAKE_MAKE_PROGRAM=mingw32-make .. -G "MinGW Makefiles"
if errorlevel 1 exit 1
mingw32-make
if errorlevel 1 exit 1
cd ..\python
"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
