# Copyright Advanced Micro Devices, Inc.
# Licensed under the Apache License Version 2.0

import os

"""
The following script is used to build a list of working features based on the unit tests
Input:
Log files generated from the tests, in the following formats: '.log', '.txt'
Output: 
Two .txt fies. The first 'feature_list.txt' that contain the functions that pass all unittests.
The second 'unsupported_feature_list.txt' that contain functions that failed even 1 of the test parameters,
or those that produced an error. 
"""

# These dictionaries will store the function names and their occurance in number of parameterized tests
pass_dict = {}
fail_dict = {}
error_dict = {}
# Debug Mode Toggle
debug = False


def find_files(directory):
    """
    Helper Function to find list of all files within a given directory
    Input: Directory path as a string
    Output: a List containing the pathnames of all files within directory
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            if debug:
                print(file_path)
    return file_paths


def write_to_file(my_dict, filename):
    """
    Helper Function to write functions strings to a .txt file with some processing to remove quotes.
    Input: Dictionary to store, and filename to store it in
    Output: a .txt file written to the file with the input filename
    """
    with open(filename, "w") as file:
        for key in my_dict.keys():
            file.write(str(key) + "\n")


def remove_first_substring(string, substring):
    """
    Helper Function to parse the names of the tests and remove specific keywords
    """
    return string.replace(substring, "", 1)


def parse_log_file(file_path):
    """
    Function to isolate individual function names from provided file
    Input: Path of log file to look for features
    Output: functions are stored to the respective dictionaries.
    """
    try:
        with open(file_path, "r") as file:
            for line in file:
                # Process each line (e.g., print, analyze, etc.)
                if "PASSED" in line or "FAILED" in line or "ERROR" in line:
                    if debug:
                        print(
                            line.strip()
                        )  # strip() removes leading/trailing whitespace, including newline characters
                    substrings = line.split("::")
                    try:
                        if len(substrings) > 1:
                            funcstrings = (
                                substrings[1]
                                .split("[")[0]
                                .split(" PASSED")[0]
                                .split(" FAILED")[0]
                                .split(" ERROR")
                            )
                        else:
                            continue
                    except Exception as e:
                        if debug:
                            print("Error String:", substrings)
                        pass
                    funcstrings = remove_first_substring(
                        funcstrings[0], "test_"
                    )
                    if debug:
                        print(funcstrings)
                    if "PASSED" in line:
                        if (
                            funcstrings in fail_dict
                            or funcstrings in error_dict
                        ):
                            continue
                        elif funcstrings not in pass_dict:
                            pass_dict[funcstrings] = 0
                        else:
                            pass_dict[funcstrings] += 1
                    if "FAILED" in line:
                        if funcstrings in pass_dict:
                            # If a single parameterized test fails for a function that passed other tests, its removed from supported list
                            pass_dict.pop(funcstrings)
                        elif funcstrings not in fail_dict:
                            fail_dict[funcstrings] = 0
                        else:
                            fail_dict[funcstrings] += 1
                    if "ERROR" in line:
                        if funcstrings in pass_dict:
                            # If a single parameterized test errors for a function that passed other tests, its removed from supported list
                            pass_dict.pop(funcstrings)
                        elif funcstrings not in error_dict:
                            error_dict[funcstrings] = 0
                        else:
                            error_dict[funcstrings] += 1

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    directory_path = "./log_files"
    files = find_files(directory_path)
    for file_path in files:
        parse_log_file(file_path)
        if debug:
            print(file_path)
    print("Total Passed:", pass_dict.keys())
    print("Total Failed:", fail_dict.keys())
    print("Total Error:", error_dict.keys())
    print("Features Passed: ", len(pass_dict))
    print("Features Unsupported: ", len(fail_dict) + len(error_dict))

    output_feature_filename = "./feature_list.txt"
    write_to_file(pass_dict, output_feature_filename)
    output_unsupported_filename = "./unsupported_feature_list.txt"
    write_to_file((fail_dict | error_dict), output_unsupported_filename)


if __name__ == "__main__":
    main()
