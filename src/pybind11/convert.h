#include <pybind11/pybind11.h>
#include <dgl/graph.h>
#include <dgl/runtime/packed_func.h>
#include <Python.h>
#include <pybind11/embed.h>

namespace py = pybind11;

using namespace dgl;
using dgl::EdgeArray;
using dgl::runtime::DGLRetValue;
using dgl::runtime::NDArray;

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
        // GraphRef is leaked
        GraphRef *holder = new GraphRef(src);
        py::module dgl = py::module::import("dgl");
        auto pbd_object = dgl.attr("_ffi").attr("_cy3").attr("core").attr("_return_pbd_object");
        py::object ph = pbd_object((long)(void *)(holder));
        Py_IncRef(ph.ptr());
        return ph;
    }
};
} // namespace detail
} // namespace pybind11

namespace pybind11
{
namespace detail
{
template <>
struct type_caster<NDArray>
{
public:
    PYBIND11_TYPE_CASTER(NDArray, _("NDArray"));

    /**
         * Conversion part 1 (Python->C++):
         */
    bool load(handle src, bool)
    {
        /* Extract PyObject from handle */
        PyObject *p0_ptr = src.ptr();
        PyObject *p_ptr = PyObject_GetAttr(p0_ptr, PyUnicode_FromString("_dgl_handle"));
        void *vhandle = PyLong_AsVoidPtr(p_ptr);
        NDArray::Container *nc = static_cast<NDArray::Container *>(vhandle);
        NDArray ndd = NDArray(nc);
        value = ndd;
        return true;
    }

    /**
         * Conversion part 2 (C++ -> Python)
         */
    static handle cast(NDArray src, return_value_policy /* policy */, handle /* parent */)
    {

        py::module dgl = py::module::import("dgl");
        DGLRetValue *ret = new DGLRetValue();
        *ret = src;
        auto make_array = dgl.attr("_ffi").attr("_cy3").attr("core").attr("_make_array");
        py::object ndarray = make_array((long)(void *)((*ret).value().v_handle), false);
        Py_IncRef(ndarray.ptr());
        return ndarray;
    }
};
} // namespace detail
} // namespace pybind11
