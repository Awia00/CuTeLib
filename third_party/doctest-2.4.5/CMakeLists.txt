cmake_minimum_required(VERSION 3.18)
project(doctest VERSION 2.4.5 LANGUAGES CXX CUDA)
include(GNUInstallDirs)

if(CUTELIB_BUILD_TESTS)
    # === Library ===
    set(LIB_NAME doctest)
    add_library(${LIB_NAME} INTERFACE)

    target_include_directories(
        ${LIB_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
endif()
