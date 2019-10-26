#include <pybind11/pybind11.h>
#include <dgl/graph.h>
#include <Python.h>
#include "convert.h"

namespace py = pybind11;

void init_ex1(py::module &);

using namespace dgl;

PYBIND11_MODULE(dglpybind, m)
{
    m.doc() = "pybind11 with dgl"; // optional module docstring
    py::module m2 = m.def_submodule("pure", "pure pybind11");
    init_ex1(m2);

    m.def("HasEdgeBetween", [](GraphPtr g, dgl_id_t src, dgl_id_t dst) {
        return g->HasEdgeBetween(src, dst);
    });

    m.def("new_gindex", [](bool multigraph) {
        GraphPtr g = Graph::Create(multigraph);
        return g;
    },
          py::arg("multigraph") = false, py::return_value_policy::move );

    m.def("HasEdgesBetween", [](GraphPtr g, IdArray src, IdArray dst) {
        return g->HasEdgesBetween(src, dst);
    });

}