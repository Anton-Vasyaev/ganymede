cmake_minimum_required(VERSION 3.19)

project(auxml.export)

set(TARGET_NAME auxml)
set(EXPORT_TARGET_NAME auxml_export)

add_custom_target(${EXPORT_TARGET_NAME})

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/../submodules/auxml/auxml
    ${CMAKE_CURRENT_BINARY_DIR}/auxml
)

# [POST BUILD] Copy dll to python native libs dir ---------------------------------------------------------------------------------------------------
# make postfix for output dir
if(WIN32)
    set(COPY_PYTHON_DIR_OS_POSTFIX win)
elseif(UNIX AND NOT APPLE)
    set(COPY_PYTHON_DIR_OS_POSTFIX linux)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(COPY_PYTHON_DIR_BIT_POSTFIX 64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(COPY_PYTHON_DIR_BIT_POSTFIX 32)
endif()

# copy native libraries
set(COPY_PYTHON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ganymede.proj/ganymede/native/libs/${COPY_PYTHON_DIR_OS_POSTFIX}${COPY_PYTHON_DIR_BIT_POSTFIX}) 

message("COPY PYTHON DIR:${COPY_PYTHON_DIR}")
add_custom_command(
    TARGET ${EXPORT_TARGET_NAME} POST_BUILD
    
    COMMAND ${CMAKE_COMMAND} -E make_directory ${COPY_PYTHON_DIR}

    COMMAND ${CMAKE_COMMAND} -E touch ${COPY_PYTHON_DIR}/__init__.py

    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:${TARGET_NAME}> ${COPY_PYTHON_DIR}/$<TARGET_FILE_NAME:${TARGET_NAME}>
)