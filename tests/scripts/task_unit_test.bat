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
SET DGL_DOWNLOAD_DIR=!CD!\_download

python -m pip install pytest psutil pandas pyyaml pydantic rdflib torchmetrics || EXIT /B 1
python -m pytest -v --junitxml=pytest_backend.xml --durations=100 tests\python\!DGLBACKEND! || EXIT /B 1
python -m pytest -v --junitxml=pytest_common.xml --durations=100 tests\python\common || EXIT /B 1
ENDLOCAL
EXIT /B
