if [ `uname` == Linux ]; then
	if [ "$PY_VER" == "3.5" ]; then
		pip install https://test-files.pythonhosted.org/packages/76/8b/21f48484938003938fbe8ff1d695ee3bb0f239b72eb48b84a5784f6c6bf6/dgl-0.0.1-cp35-cp35m-manylinux1_x86_64.whl
	elif [ "$PY_VER" == "3.6" ]; then
		pip install https://test-files.pythonhosted.org/packages/41/42/9c156ba41de69812a720ba28838654fc5e7572ca9acb46c95f942d555192/dgl-0.0.1-cp36-cp36m-manylinux1_x86_64.whl
	fi
elif [ `uname` == Darwin ]; then
	echo OSX is currently not supported
	exit 1
fi
