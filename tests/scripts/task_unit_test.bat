@ECHO OFF
SETLOCAL EnableDelayedExpansion

IF x%1x==xx (
	ECHO Specify backend
	EXIT /B 1
) ELSE (
	SET BACKEND=%1
)
CALL workon %BUILD_TAG%

SET PYTHONPATH=tests;!CD!\python;!PYTHONPATH!
SET DGLBACKEND=!BACKEND!
SET DGL_LIBRARY_PATH=!CD!\build
SET DGL_DOWNLOAD_DIR=!CD!

python -m nose -v --with-xunit tests\!DGLBACKEND! || EXIT /B 1
python -m nose -v --with-xunit tests\graph_index || EXIT /B 1
python -m nose -v --with-xunit tests\compute || EXIT /B 1
ENDLOCAL
EXIT /B
