
@echo on
set path=D:\Anaconda3;D:\Anaconda3\Library\bin;D:\Anaconda3\Scripts;D:\Anaconda3\condabin;%path%
set path=C:\Users\Administrator\anaconda3;C:\Users\Administrator\anaconda3\Library\bin;%path%
set path=C:\Users\Administrator\anaconda3\Scripts;C:\Users\Administrator\anaconda3\condabin;%path%
set path=C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Library\bin;%path%
set path=C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3\condabin;%path%
path

python -m build

twine check dist/*

twine upload dist/*

rd /s /Q dist

rd /s /Q mgetool.egg-info

pause

pause

exit