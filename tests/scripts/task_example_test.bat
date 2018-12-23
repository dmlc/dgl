@ECHO OFF
SETLOCAL EnableDelayedExpansion

IF x%1x==xx (
	ECHO Must supply CPU or GPU
	GOTO :FAIL
) ELSE IF x%1x==xCPUx (
	SET DEV=-1
) ELSE IF x%1x==xGPUx (
	SET DEV=0
	SET CUDA_VISIBLE_DEVICES=0
) ELSE (
	ECHO Must supply CPU or GPU
	GOTO :FAIL
)

PUSHD ..\..\examples\pytorch
python pagerank.py || GOTO :FAIL
python gcn\gcn.py --dataset cora --gpu !dev! || GOTO :FAIL
python gcn\gcn_spmv.py --dataset cora --gpu !dev! || GOTO :FAIL
POPD
ENDLOCAL
EXIT /B

:FAIL
ECHO Example test failed
ENDLOCAL
EXIT /B 1
