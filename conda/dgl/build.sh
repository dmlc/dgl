mkdir build
cd build
cmake ..
make
cd ../python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
