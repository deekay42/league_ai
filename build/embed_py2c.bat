REM this script needs to be given the path to the pythondir. there it'll create a folder named dist with all required dlls and libs
REM it will also create cython.h in the include dir in the pythondir
if [%1]==[] goto usage
if [%2]==[] goto usage

set PATH=C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\x64;%PATH%
call vcvars64.bat
set INIT_DIR=%cd%
CALL :NORMALIZEPATH %~1 
set PYTHONDIR=%RETVAL%
CALL :NORMALIZEPATH %~2
set OUT_DIR=%RETVAL%
echo %PYTHONDIR%
echo %OUT_DIR%
REM rmdir /s /q .\dist
cd %PYTHONDIR%
rmdir /s /q .\tmp_build
mkdir tmp_build


python -m PyInstaller -d noarchive main.py --distpath tmp_build
MOVE tmp_build\main\*.dll tmp_build
del /Q /S .\tmp_build\main\*.exe
rmdir /s /q .\tmp_build\main\utils
rmdir /s /q .\tmp_build\main\train_model
rmdir /s /q .\tmp_build\main\constants

mkdir tmp_build\tmp
python "build/setup.py" build_ext --build-lib tmp_build\tmp
del /Q /S .\tmp_build\tmp\*.c
del /Q /S .\tmp_build\tmp\*.pyc
ROBOCOPY /NFL /NDL  tmp_build\tmp tmp_build\main *.* /S /MOVE
MOVE tmp_build\main tmp_build\py_libs
echo #cython: language_level=3 >cython_main.pyx
echo import os; os.environ["TCL_LIBRARY"]="py_libs\\tcl"; os.environ["TK_LIBRARY"]="py_libs\\tk"; import sys; sys.path = ["py_libs/base_library.zip", "py_libs"]; sys.argv = ["cython_main.pyx"]; import main; m = main.Main(); m.run()>>cython_main.pyx
cython cython_main.pyx --embed
del /Q cython_main.pyx

sed -i "s#int wmain.*#int runCythonCode(){  int argc = 0;  wchar_t** argv = nullptr; Py_SetPath(L\"py_libs;py_libs/base_library.zip\");#" cython_main.c
move cython_main.c "%INIT_DIR%\include\cython.h"
SET src_folder=tmp_build
SET tar_folder=%OUT_DIR%
ROBOCOPY /NFL /NDL %src_folder% %tar_folder% *.* /S /MOVE
cd %INIT_DIR%
SET src_folder=%PYTHONDIR%\models\best
SET tar_folder=%OUT_DIR%\models\best
ROBOCOPY /NFL /NDL  %src_folder% %tar_folder% *.* /S
del /Q /S %OUT_DIR%\models\*accuracies*

mkdir %OUT_DIR%\assets\data
COPY %PYTHONDIR%\..\assets\data\champ2id.json %OUT_DIR%\assets\data
COPY %PYTHONDIR%\..\assets\data\item2id.json %OUT_DIR%\assets\data
COPY %PYTHONDIR%\..\assets\data\self2id.json %OUT_DIR%\assets\data
ROBOCOPY /NFL /NDL  %PYTHONDIR%\..\assets\fonts %OUT_DIR%\assets\fonts *.* /S
ROBOCOPY /NFL /NDL  %PYTHONDIR%\..\assets\icons %OUT_DIR%\assets\icons *.* /S
ROBOCOPY /NFL /NDL  %PYTHONDIR%\..\assets\imgs %OUT_DIR%\assets\imgs *.* /S
ROBOCOPY /NFL /NDL  %PYTHONDIR%\..\assets\item_imgs %OUT_DIR%\assets\item_imgs *.* /S

exit /B

:usage
@echo Usage: Need to specify 1:pythondir 2:out_dir
exit /B 1

:NORMALIZEPATH
  SET RETVAL=%~dpfn1
  EXIT /B