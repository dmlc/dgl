#!/usr/bin/env python3
# pylint: disable=protected-access, unused-variable, locally-disabled, len-as-condition
"""Lint helper to generate lint summary of source.

Copyright by Contributors.

Borrowed from dmlc-core/scripts/lint.py@939c052
"""
from __future__ import print_function

import argparse
import codecs
import os
import re
import sys

import cpplint
from cpplint import _cpplint_state
from pylint import epylint

CXX_SUFFIX = set(["cc", "c", "cpp", "h", "cu", "hpp", "cuh"])
PYTHON_SUFFIX = set(["py"])


def filepath_enumerate(paths):
    """Enumerate the file paths of all subfiles of the list of paths"""
    out = []
    for path in paths:
        if os.path.isfile(path):
            out.append(path)
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    out.append(os.path.normpath(os.path.join(root, name)))
    return out


# pylint: disable=useless-object-inheritance
class LintHelper(object):
    """Class to help runing the lint and records summary"""

    @staticmethod
    def _print_summary_map(strm, result_map, ftype):
        """Print summary of certain result map."""
        if len(result_map) == 0:
            return 0
        npass = sum(1 for x in result_map.values() if len(x) == 0)
        strm.write(
            f"====={npass}/{len(result_map)} {ftype} files passed check=====\n"
        )
        for fname, emap in result_map.items():
            if len(emap) == 0:
                continue
            strm.write(
                f"{fname}: {sum(emap.values())} Errors of {len(emap)} Categories map={str(emap)}\n"
            )
        return len(result_map) - npass

    def __init__(self):
        self.project_name = None
        self.cpp_header_map = {}
        self.cpp_src_map = {}
        self.python_map = {}
        pylint_disable = [
            "superfluous-parens",
            "too-many-instance-attributes",
            "too-few-public-methods",
        ]
        # setup pylint
        self.pylint_opts = [
            "--extension-pkg-whitelist=numpy",
            "--disable=" + ",".join(pylint_disable),
        ]

        self.pylint_cats = set(["error", "warning", "convention", "refactor"])
        # setup cpp lint
        cpplint_args = [
            "--quiet",
            "--extensions=" + (",".join(CXX_SUFFIX)),
            ".",
        ]
        _ = cpplint.ParseArguments(cpplint_args)
        cpplint._SetFilters(
            ",".join(
                [
                    "-build/c++11",
                    "-build/namespaces",
                    "-build/include,",
                    "+build/include_what_you_use",
                    "+build/include_order",
                ]
            )
        )
        cpplint._SetCountingStyle("toplevel")
        cpplint._line_length = 80

    def process_cpp(self, path, suffix):
        """Process a cpp file."""
        _cpplint_state.ResetErrorCounts()
        cpplint.ProcessFile(str(path), _cpplint_state.verbose_level)
        _cpplint_state.PrintErrorCounts()
        errors = _cpplint_state.errors_by_category.copy()

        if suffix == "h":
            self.cpp_header_map[str(path)] = errors
        else:
            self.cpp_src_map[str(path)] = errors

    def process_python(self, path):
        """Process a python file."""
        (pylint_stdout, pylint_stderr) = epylint.py_run(
            " ".join([str(path)] + self.pylint_opts), return_std=True
        )
        emap = {}
        err = pylint_stderr.read()
        if len(err):
            print(err)
        for line in pylint_stdout:
            sys.stderr.write(line)
            key = line.split(":")[-1].split("(")[0].strip()
            if key not in self.pylint_cats:
                continue
            if key not in emap:
                emap[key] = 1
            else:
                emap[key] += 1
        self.python_map[str(path)] = emap

    def print_summary(self, strm):
        """Print summary of lint."""
        nerr = 0
        nerr += LintHelper._print_summary_map(
            strm, self.cpp_header_map, "cpp-header"
        )
        nerr += LintHelper._print_summary_map(
            strm, self.cpp_src_map, "cpp-source"
        )
        nerr += LintHelper._print_summary_map(strm, self.python_map, "python")
        if nerr == 0:
            strm.write("All passed!\n")
        else:
            strm.write(f"{nerr} files failed lint\n")
        return nerr


# singleton helper for lint check
_HELPER = LintHelper()


def get_header_guard_dmlc(filename):
    """Get Header Guard Convention for DMLC Projects.

    For headers in include, directly use the path
    For headers in src, use project name plus path

    Examples: with project-name = dmlc
        include/dmlc/timer.h -> DMLC_TIMTER_H_
        src/io/libsvm_parser.h -> DMLC_IO_LIBSVM_PARSER_H_
    """
    fileinfo = cpplint.FileInfo(filename)
    file_path_from_root = fileinfo.RepositoryName()
    inc_list = ["include", "api", "wrapper", "contrib"]
    if os.name == "nt":
        inc_list.append("mshadow")

    if (
        file_path_from_root.find("src/") != -1
        and _HELPER.project_name is not None
    ):
        idx = file_path_from_root.find("src/")
        file_path_from_root = (
            _HELPER.project_name + file_path_from_root[idx + 3 :]
        )
    else:
        idx = file_path_from_root.find("include/")
        if idx != -1:
            file_path_from_root = file_path_from_root[idx + 8 :]
        for spath in inc_list:
            prefix = spath + "/"
            if file_path_from_root.startswith(prefix):
                file_path_from_root = re.sub(
                    "^" + prefix, "", file_path_from_root
                )
                break
    return re.sub(r"[-./\s]", "_", file_path_from_root).upper() + "_"


cpplint.GetHeaderGuardCPPVariable = get_header_guard_dmlc


def process(fname, allow_type):
    """Process a file."""
    fname = str(fname)
    arr = fname.rsplit(".", 1)
    if fname.find("#") != -1 or arr[-1] not in allow_type:
        return
    if arr[-1] in CXX_SUFFIX:
        _HELPER.process_cpp(fname, arr[-1])
    if arr[-1] in PYTHON_SUFFIX:
        _HELPER.process_python(fname)


def main():
    """Main entry function."""
    parser = argparse.ArgumentParser(description="lint source codes")
    parser.add_argument("project", help="project name")
    parser.add_argument(
        "filetype", choices=["python", "cpp", "all"], help="source code type"
    )
    parser.add_argument("path", nargs="+", help="path to traverse")
    parser.add_argument(
        "--exclude_path",
        nargs="+",
        default=[],
        help="exclude this path, and all subfolders if path is a folder",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="run cpplint in quiet mode"
    )
    parser.add_argument("--pylint-rc", default=None, help="pylint rc file")
    args = parser.parse_args()

    _HELPER.project_name = args.project
    if args.pylint_rc is not None:
        _HELPER.pylint_opts = [
            "--rcfile=" + args.pylint_rc,
        ]
    file_type = args.filetype
    allow_type = []
    if file_type in ("python", "all"):
        allow_type += PYTHON_SUFFIX
    if file_type in ("cpp", "all"):
        allow_type += CXX_SUFFIX
    allow_type = set(allow_type)
    if sys.version_info.major == 2 and os.name != "nt":
        sys.stderr = codecs.StreamReaderWriter(
            sys.stderr,
            codecs.getreader("utf8"),
            codecs.getwriter("utf8"),
            "replace",
        )
    # get excluded files
    excluded_paths = filepath_enumerate(args.exclude_path)
    for path in args.path:
        if os.path.isfile(path):
            normpath = os.path.normpath(path)
            if normpath not in excluded_paths:
                process(path, allow_type)
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    file_path = os.path.normpath(os.path.join(root, name))
                    if file_path not in excluded_paths:
                        process(file_path, allow_type)
    nerr = _HELPER.print_summary(sys.stderr)
    sys.exit(nerr > 0)


if __name__ == "__main__":
    main()
