@ECHO OFF
SETLOCAL EnableDelayedExpansion

IF x%1x==xx (
	ECHO Specify backend
	EXIT /B 1
) ELSE (
	SET BACKEND=%1
)

SET PYTHONPATH=tests;!PYTHONPATH!

python -m nose -v --with-xunit tests\!BACKEND! || EXIT /B 1
python -m nose -v --with-xunit tests\graph_index || EXIT /B 1
python -m nose -v --with-xunit tests\compute || EXIT /B 1
EXIT /B
