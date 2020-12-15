@ECHO OFF
SETLOCAL EnableDelayedExpansion

IF NOT x%2x==xx (
	CALL conda activate %2 || EXIT /B 1
	python %1 || EXIT /B 1
	CALL conda deactivate || EXIT /B 1
) ELSE (
	python %1 || EXIT /B 1
)

ENDLOCAL
EXIT /B
