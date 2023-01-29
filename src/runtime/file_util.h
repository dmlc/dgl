/**
 *  Copyright (c) 2017 by Contributors
 * @file file_util.h
 * @brief Minimum file manipulation util for runtime.
 */
#ifndef DGL_RUNTIME_FILE_UTIL_H_
#define DGL_RUNTIME_FILE_UTIL_H_

#include <string>
#include <unordered_map>

#include "meta_data.h"

namespace dgl {
namespace runtime {
/**
 * @brief Get file format from given file name or format argument.
 * @param file_name The name of the file.
 * @param format The format of the file.
 */
std::string GetFileFormat(
    const std::string& file_name, const std::string& format);

/**
 * @return the directory in which DGL stores cached files.
 *         May be set using DGL_CACHE_DIR; defaults to system locations.
 */
std::string GetCacheDir();

/**
 * @brief Get meta file path given file name and format.
 * @param file_name The name of the file.
 */
std::string GetMetaFilePath(const std::string& file_name);

/**
 * @brief Get file basename (i.e. without leading directories)
 * @param file_name The name of the file.
 * @return the base name
 */
std::string GetFileBasename(const std::string& file_name);

/**
 * @brief Load binary file into a in-memory buffer.
 * @param file_name The name of the file.
 * @param data The data to be loaded.
 */
void LoadBinaryFromFile(const std::string& file_name, std::string* data);

/**
 * @brief Load binary file into a in-memory buffer.
 * @param file_name The name of the file.
 * @param data The binary data to be saved.
 */
void SaveBinaryToFile(const std::string& file_name, const std::string& data);

/**
 * @brief Save meta data to file.
 * @param file_name The name of the file.
 * @param fmap The function info map.
 */
void SaveMetaDataToFile(
    const std::string& file_name,
    const std::unordered_map<std::string, FunctionInfo>& fmap);

/**
 * @brief Load meta data to file.
 * @param file_name The name of the file.
 * @param fmap The function info map.
 */
void LoadMetaDataFromFile(
    const std::string& file_name,
    std::unordered_map<std::string, FunctionInfo>* fmap);
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_RUNTIME_FILE_UTIL_H_
