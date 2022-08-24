git submodule init
git submodule update
mkdir build
mkdir -p build/tensoradapter/pytorch

if [[ "$CACHEDIR" = "" ]]; then
	echo No prebuilt binary directory specified, building with default options...
	cd build
	cmake -DUSE_OPENMP=ON -DUSE_CUDA=$USE_CUDA -DBUILD_TORCH=ON ..
	make -j4
	cd ..
else
	if [[ "$OSTYPE" = "darwin"* ]]; then
		echo Copying prebuilt CPU binary...
		cp $CACHEDIR/libdgl.dylib build
		if [[ -e $CACHEDIR/tensoradapter_pytorch_cpu ]]; then
			cp -v $CACHEDIR/tensoradapter_pytorch_cpu/*.dylib build/tensoradapter/pytorch
		fi
	else
		if [[ $USE_CUDA = 'ON' ]]; then
			echo Copying prebuilt CUDA $CUDA_VER binary...
			cp -v $CACHEDIR/libdgl.so.cu$CUDA_VER build/libdgl.so
			if [[ -e $CACHEDIR/tensoradapter_pytorch_cu$CUDA_VER ]]; then
				cp -v $CACHEDIR/tensoradapter_pytorch_cu$CUDA_VER/*.so build/tensoradapter/pytorch
			fi
		else
			echo Copying prebuilt CPU binary...
			cp -v $CACHEDIR/libdgl.so.cpu build/libdgl.so
			if [[ -e $CACHEDIR/tensoradapter_pytorch_cpu ]]; then
				cp -v $CACHEDIR/tensoradapter_pytorch_cpu/*.so build/tensoradapter/pytorch
			fi
		fi
	fi
fi
cd python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
rm -f ../build/libdgl.so ../build/tensoradapter/pytorch/*.so
