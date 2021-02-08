cmake_minimum_required(VERSION 3.15)
project(CuTeLib VERSION 0.0.1 LANGUAGES CXX CUDA)
include(GNUInstallDirs)

# === Library === 
set(LIB_NAME cutelib)
add_library(${LIB_NAME} INTERFACE)

target_include_directories(
    ${LIB_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


# == Install ==
install(
    TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}-config
)

install(
    EXPORT ${LIB_NAME}-config
    NAMESPACE ${LIB_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${LIB_NAME}
)

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    TARGETS ${LIB_NAME}
    EXPORT ${LIB_NAME}_Targets
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)


# === TESTS === 
set(TEST_NAME cutelib_tests)

file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp" "tests/*.cu")
add_executable(${TEST_NAME} ${TEST_SOURCES})
target_link_libraries(${TEST_NAME} PRIVATE ${LIB_NAME} doctest)

set_target_properties(
    ${TEST_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)


# == Install ==
install(
    TARGETS ${TEST_NAME}
    EXPORT ${TEST_NAME}-config)

install(
    EXPORT ${TEST_NAME}-config
    NAMESPACE ${LIB_NAME}::tests
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${TEST_NAME})


# === Examples === 
set(EXAMPLES_NAME cutelib_examples)

file(GLOB_RECURSE EXAMPLE_SOURCES "examples/*.cpp" "examples/*.cu")
add_executable(${EXAMPLES_NAME} ${EXAMPLE_SOURCES})
target_link_libraries(${EXAMPLES_NAME} PRIVATE ${LIB_NAME})

set_target_properties(
    ${EXAMPLES_NAME}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

# == Install ==
install(
    TARGETS ${EXAMPLES_NAME}
    EXPORT ${EXAMPLES_NAME}-config)

install(
    EXPORT ${EXAMPLES_NAME}-config
    NAMESPACE ${LIB_NAME}::examples
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${EXAMPLES_NAME})
