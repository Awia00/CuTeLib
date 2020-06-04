cmake_minimum_required(VERSION 3.15)
project(CuTeLib VERSION 0.0.1 LANGUAGES CXX CUDA)
include(GNUInstallDirs)

# === Library === 
set(LIB_NAME CuTeLib)
add_library(${LIB_NAME} INTERFACE)

target_include_directories(
    ${LIB_NAME} INTERFACE
    $<BUILD_INTERFACE:${${LIB_NAME}_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(${LIB_NAME} INTERFACE cxx_std_17)


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
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}
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
set(TEST_NAME CuTeLibTests)
add_executable(${TEST_NAME} tests/main.tests.cu)
target_link_libraries(${TEST_NAME} PRIVATE ${LIB_NAME})

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

install(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${TEST_NAME})