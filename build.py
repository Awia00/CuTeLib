import subprocess


def run():
    try:
        build_type = "Debug"
        build_folder = "build"
        install_folder = "inst"

        cmake_cmd = [
            "cmake",
            f"-DCMAKE_INSTALL_PREFIX={install_folder}",
            "-S",
            ".",
            "-B",
            build_folder,
        ]
        build_cmd = [
            "cmake",
            "--build",
            build_folder,
            "--config",
            build_type,
        ]
        install_cmd = [
            "cmake",
            "--install",
            build_folder,
            "--config",
            build_type,
        ]
        subprocess.check_call(cmake_cmd)
        subprocess.check_call(build_cmd)
        subprocess.check_call(install_cmd)
    except subprocess.CalledProcessError as err:
        print(err)
        raise


if __name__ == "__main__":
    run()