cmake_minimum_required(VERSION 2.8)
project(gann)

find_package(Threads)

set(GANN_SRC src/gann.cpp)
set(GANN_HDR include/gann/gann.hpp)

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

add_executable(paraboloid examples/paraboloid/src/paraboloid.cpp)
configure_file(examples/paraboloid/scripts/paraboloid_stats.gpi paraboloid_stats.gpi COPYONLY)
configure_file(examples/paraboloid/scripts/paraboloid_run.sh paraboloid_run.sh COPYONLY)

add_executable(pointsfit examples/pointsfit/src/pointsfit.cpp)
configure_file(examples/pointsfit/scripts/pointsfit_stats.gpi pointsfit_stats.gpi COPYONLY)
configure_file(examples/pointsfit/scripts/pointsfit_func.gpi pointsfit_func.gpi COPYONLY)
configure_file(examples/pointsfit/scripts/pointsfit_run.sh pointsfit_run.sh COPYONLY)

target_link_libraries(paraboloid ${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(pointsfit ${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})

