cmake_minimum_required(VERSION 2.8)
project(gann)

find_package(Threads)

set(GANN_SRC src/ga.cpp src/mlp.cpp src/util.cpp)
set(GANN_HDR include/gann/ga.h include/gann/mlp.h include/gann/util.h)

include_directories(include)

add_library(${PROJECT_NAME} SHARED ${GANN_SRC})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(FILES ${GANN_HDR} DESTINATION include/gann)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/include/gann")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxy/Doxyfile.in
                   ${CMAKE_CURRENT_BINARY_DIR}/doxy/Doxyfile @ONLY)
    add_custom_target(doc
                      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxy/Doxyfile
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      COMMENT "Generating documentation" VERBATIM)
endif(DOXYGEN_FOUND)

add_executable(paraboloid examples/paraboloid/main.cpp)
target_link_libraries(paraboloid ${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})

