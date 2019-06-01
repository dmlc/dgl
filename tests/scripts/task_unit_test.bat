@ECHO OFF
SETLOCAL EnableDelayedExpansion

IF x%1x==xx (
	ECHO Specify backend
	EXIT /B 1
) ELSE (
	SET DGLTESTDEV=%1
)

SET PYTHONPATH=tests;!PYTHONPATH!

python -m nose -v --with-xunit tests\!DGLBACKEND! || EXIT /B 1
python -m nose -v --with-xunit tests\graph_index || EXIT /B 1
python -m nose -v --with-xunit tests\compute || EXIT /B 1
ENDLOCAL
EXIT /B
