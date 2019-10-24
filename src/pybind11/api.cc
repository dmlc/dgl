#include <pybind11/pybind11.h>
#include <dgl/graph.h>
#include <Python.h>

namespace py = pybind11;

void init_ex1(py::module &);

using namespace dgl;

namespace pybind11
{
namespace detail
{
template <>
struct type_caster<GraphPtr>
{
public:
    PYBIND11_TYPE_CASTER(GraphPtr, _("GraphPtr"));

    /**
         * Conversion part 1 (Python->C++):
         */
    bool load(handle src, bool)
    {
        /* Extract PyObject from handle */
        PyObject *p0_ptr = src.ptr();
        PyObject *p_ptr = PyObject_GetAttr(p0_ptr, PyUnicode_FromString("handle"));
        if (!PyObject_HasAttr(p_ptr, PyUnicode_FromString("value")))
        {
            return false;
        }
        PyObject *ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
        if (ptr_as_int == Py_None)
        {
            return false;
        }
        void *ptr = PyLong_AsVoidPtr(ptr_as_int);
        dgl::GraphRef *ad = static_cast<dgl::GraphRef *>(ptr);
        value = (*ad).sptr();
        return true;
    }

    /**
         * Conversion part 2 (C++ -> Python)
         */
    static handle cast(GraphPtr src, return_value_policy /* policy */, handle /* parent */)
    {
        GraphRef *holder = new GraphRef(src);
        return PyLong_FromVoidPtr((void *)(holder));
    }
};
} // namespace detail
} // namespace pybind11

PYBIND11_MODULE(dglpybind, m)
{
    m.doc() = "pybind11 with dgl"; // optional module docstring

    init_ex1(m);

    m.def("ptr", [](GraphPtr g) {
        return g->HasEdgeBetween(0, 1);
    });

    m.def("new_gindex", [](bool multigraph) {
        GraphPtr g = Graph::Create(multigraph);
        return g;
    },
          py::arg("multigraph") = false);
}