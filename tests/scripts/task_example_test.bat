@ECHO OFF
SETLOCAL EnableDelayedExpansion

SET GCN_EXAMPLE_DIR=.\examples\pytorch

IF x%1x==xx (
	ECHO Must supply CPU or GPU
	GOTO :FAIL
) ELSE IF x%1x==xcpux (
	SET DEV=-1
) ELSE IF x%1x==xgpux (
	SET DEV=0
	SET CUDA_VISIBLE_DEVICES=0
) ELSE (
	ECHO Must supply CPU or GPU
	GOTO :FAIL
)
CALL workon %BUILD_TAG%

SET DGLBACKEND=pytorch
SET DGL_LIBRARY_PATH=!CD!\build
SET PYTHONPATH=!CD!\python;!PYTHONPATH!
SET DGL_DOWNLOAD_DIR=!CD!

PUSHD !GCN_EXAMPLE_DIR!
python pagerank.py || GOTO :FAIL
python gcn\gcn.py --dataset cora --gpu !DEV! || GOTO :FAIL
python gcn\gcn_spmv.py --dataset cora --gpu !DEV! || GOTO :FAIL
POPD
ENDLOCAL
EXIT /B

:FAIL
ECHO Example test failed
ENDLOCAL
EXIT /B 1
