@echo off

cls
call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64

cl^
 /DSDL_MAIN_USE_CALLBACKS=1^
 /MD^
 /ZI^
 /Ox^
 /std:c++20^
 .\play\core\*.cpp^
 .\play\os\*.cpp^
 .\play_sdl3\*.cpp^
 .\examples\*.cpp^
 /I.\^
 /I.\3rdparty\SDL3\include\^
 /Fe.\bin\play_sdl3_msvc.exe^
 /Fd.\bin\play_sdl3_msvc^
 /Fo.\bin\^
 /link /DEBUG:FULL /LIBPATH:./3rdparty/SDL3/lib/x64/ /SUBSYSTEM:CONSOLE SDL3.lib
