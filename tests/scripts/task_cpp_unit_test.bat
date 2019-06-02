@ECHO OFF
SETLOCAL EnableDelayedExpansion

PUSHD build
runUnitTests.exe || EXIT /B 1
POPD
