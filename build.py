"""
Builds and installs the project
"""

import subprocess
import os


def build_cpp():
    print("Starting build_cpp...")
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
    print("build_cpp success\n")


def build_docs(doc_path: str):
    print("Starting build_docs...")
    try:
        subprocess.call(["python", f"{doc_path}/build.py"])
    except subprocess.CalledProcessError as err:
        print(err)
        raise
    print("build_docs success\n")


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Build the cutelib wiki')
    parser.add_argument('-d',
                        '--doc_path',
                        default="../CuTeLib.wiki",
                        help="path to the doc repo")

    return parser.parse_args()


def ensure_arguments(args):
    assert os.path.exists(args.doc_path), "code_path did not exist"


def run():
    args = parse_arguments()
    ensure_arguments(args)

    build_cpp()
    build_docs(args.doc_path)


if __name__ == "__main__":
    run()