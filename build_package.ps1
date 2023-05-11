# load conan dependencies
cd submodules/auxml
conan_update_x64.bat
cd ../..


# configuration cmake project
Start-Process -FilePath 'cmake' -Wait -NoNewWindow -ArgumentList `
    '--no-warn-unused-cli',
    '-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE',
    '-DCMAKE_BUILD_TYPE:STRING=Release',
    '-S auxml.export.proj',
    '-B auxml.export.proj/build'


# build cmake target and export native libraries to ganymede.proj
Start-Process -FilePath 'cmake' -Wait -NoNewWindow -ArgumentList `
    '--build auxml.export.proj/build',
    '--config Release',
    '--target auxml_export' 


# build pip package
python -m setup bdist_wheel