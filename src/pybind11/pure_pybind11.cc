#include <pybind11/pybind11.h>
#include <dgl/graph.h>
#include <dgl/immutable_graph.h>
#include "convert.h"

namespace py = pybind11;

void init_ex1(py::module &m)
{
    py::class_<dgl::ImmutableGraph, dgl::ImmutableGraphPtr>(m, "ImmutableGraph")
        .def("has_edge_between", &dgl::ImmutableGraph::HasEdgeBetween);
    py::class_<EdgeArray>(m, "EdgeArray")
        .def_readonly("src", &dgl::EdgeArray::src)
        .def_readonly("dst", &dgl::EdgeArray::dst)
        .def_readonly("id", &dgl::EdgeArray::id);
    py::class_<dgl::Graph, dgl::MutableGraphPtr>(m, "MutableGraph")
        .def(py::init<bool>(), py::arg("multigraph") = false)
        .def("has_edges_between", &dgl::Graph::HasEdgesBetween)
        .def("has_edge_between", &dgl::Graph::HasEdgeBetween)
        .def("add_nodes", &dgl::Graph::AddVertices)
        .def("add_edges", &dgl::Graph::AddEdges)
        .def("edges", &dgl::Graph::Edges, py::arg("order") = "srcdst");
}