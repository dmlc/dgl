# Find the METIS includes and library
#
# This module defines
#  METIS_INCLUDE_DIR        -    where to find metis.h
#  METIS_LIBRARIES          -    libraries to link against to use METIS.
#  METIS_FOUND              -    METIS library was found

INCLUDE(FindPackageHandleStandardArgs)

FIND_PATH(METIS_INCLUDE_DIR
    NAMES
    "metis.h"
    PATHS
    ${EXTERNAL_METIS_PATH}
    )


FIND_LIBRARY(METIS_LIBRARIES
    NAMES
    libmetis metis
    PATHS
    ${EXTERNAL_METIS_LIB_PATH}
    )


FIND_PACKAGE_HANDLE_STANDARD_ARGS(METIS DEFAULT_MSG METIS_INCLUDE_DIR METIS_LIBRARIES)
MARK_AS_ADVANCED(METIS_LIBRARIES METIS_INCLUDE_DIR)
