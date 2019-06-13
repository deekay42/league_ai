REM this script needs to be given the path to the pythondir. there it'll create a folder named dist with all required dlls and libs
REM it will also create cython.h in the include dir in the pythondir
set PATH=C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\x64;%PATH%
call vcvars64.bat
set INIT_DIR=%cd%
set PYTHONDIR=%~1
set OUT_DIR=%~2
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
echo import os; print("\ncurrent working dir:" + os.getcwd()); import sys; sys.path = ["py_libs/base_library.zip", "py_libs"]; print(sys.path); import main; m = main.Main(); m.run()>>cython_main.pyx
cython cython_main.pyx --embed
del /Q cython_main.pyx

sed -i "s/int wmain.*/int runCythonCode(){  int argc = 0;  wchar_t** argv = nullptr;/" cython_main.c
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
ROBOCOPY /NFL /NDL  %PYTHONDIR%\..\assets\icon %OUT_DIR%\assets\icon *.* /S
ROBOCOPY /NFL /NDL  %PYTHONDIR%\..\assets\imgs %OUT_DIR%\assets\imgs *.* /S
ROBOCOPY /NFL /NDL  %PYTHONDIR%\..\assets\item_imgs %OUT_DIR%\assets\item_imgs *.* /S