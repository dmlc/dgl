/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/bipartite.h
 * \brief Bipartite graph
 */

#ifndef DGL_BIPARTITE_H_
#define DGL_BIPARTITE_H_

#include "heterograph_interface.h"

namespace dgl {

// forward declaration
class COOBipartite;
class CSRBipartite;
typedef std::shared_ptr<COOBipartite> COOPtr;
typedef std::shared_ptr<CSRBipartite> CSRPtr;

class BipartiteGraph : public HeteroGraphInterface {
 public:
};

};  // namespace dgl

#endif  // DGL_BIPARTITE_H_
