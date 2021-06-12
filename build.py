"""
Builds and installs the project
"""

import os
import shutil
import subprocess
import time


def ensure_build_type(build_type):
    values = {
        "debug": "Debug",
        "release": "Release",
        "relwithdebinfo": "RelWithDebInfo"
    }
    return values[build_type.lower()]


def build_cpp(args):
    print("Starting build_cpp...")

    try:
        build_type = ensure_build_type(args.build_type)
        build_folder = "build"
        install_folder = "inst"

        if args.fresh:
            time.sleep(1)
            shutil.rmtree(build_folder, ignore_errors=True)
            shutil.rmtree(install_folder, ignore_errors=True)

        cmake_cmd = [
            "cmake",
            f"-DCMAKE_INSTALL_PREFIX={install_folder}",
            f"-DCUTELIB_BUILD_TESTS=ON",
            f"-DCUTELIB_BUILD_EXAMPLES=ON",
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

        print(f"Running: {cmake_cmd}")
        subprocess.check_call(cmake_cmd)

        print(f"Running: {build_cmd}")
        subprocess.check_call(build_cmd)

        print(f"Running: {install_cmd}")
        subprocess.check_call(install_cmd)
    except subprocess.CalledProcessError as err:
        print(err)
        raise err
    print("build_cpp success\n")


def build_docs(doc_path: str):
    print("Starting build_docs...")
    try:
        assert os.path.exists(doc_path), "code_path did not exist"
        subprocess.call(["python", f"{doc_path}/build.py"])
    except subprocess.CalledProcessError as err:
        print(err)
        raise err
    print("build_docs success\n")


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Build the cutelib wiki')
    parser.add_argument('-f',
                        '--fresh',
                        action='store_true',
                        help='Cleanup binary and temporary folders')
    parser.add_argument('--build_type', default='Debug', help='Build Type')
    parser.add_argument('-d', '--doc_path', help="Path to the doc repo")

    return parser.parse_args()


def run():
    args = parse_arguments()

    build_cpp(args)
    if args.doc_path:
        build_docs(args.doc_path)


# For official build run with: python build.py --build_type=Release --fresh --doc_path=../CuTeLib.wiki
if __name__ == "__main__":
    run()
