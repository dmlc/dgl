.. currentmodule:: dgl

DGL Foreign Function Interface (FFI)
====================================

We all like Python because it is easy to manipulate. We all like C because it
is fast, reliable and typed. To have the merits of both ends, DGL is mostly in
python, for quick prototyping, while lowers the performance-critical part to C.
Thus, DGL developers frequently face the scenario to write a C routine and has
it exposed to python, via a mechanism called *Foreign Function Interface (FFI)*.

There are many FFI solutions out there. In DGL, we want to keep it simple,
intuitive and efficient for critical use cases. That's why when we came across the
FFI solution in the TVM project, we immediately fell for it. It exploits the idea of
functional programming so that it exposes only a dozens of C APIs and new APIs
can be built upon it.

We decided to borrow the idea (shamelessly). For example, to define a C
API that is exposed to python is only a few lines of codes:

.. code:: c++

   // file: calculator.cc (put it in dgl/src folder)
   #include <dgl/runtime/packed_func.h>
   #include <dgl/runtime/registry.h>

   using namespace dgl::runtime;

   DGL_REGISTER_GLOBAL("calculator.MyAdd")
   .set_body([] (DGLArgs args, DGLRetValue* rv) {
       int a = args[0];
       int b = args[1];
       *rv = a + b;
     });

Compile and build the library. On the python side, create a
``calculator.py`` file under ``dgl/python/dgl/``

.. code:: python

   # file: calculator.py
   from ._ffi.function import _init_api

   def add(a, b):
     # MyAdd has been registered via `_ini_api` call below
     return MyAdd(a, b)

   _init_api("dgl.calculator")

The trick is that the FFI system first masks the type information of the
function arguments, so all the C function calls can go through one C API
(``DGLFuncCall``). The type information is retrieved in the function body by
static conversion, and we will do runtime type check to make sure that the type
conversion is correct. The overhead of such back-and-forth is negligible as
long as the function call is not too light (the above example is actually a bad
one). TVM's `PackedFunc
document <https://docs.tvm.ai/dev/runtime.html#packedfunc>`_ has more details.

Defining new types
------------------

``DGLArgs`` and ``DGLRetValue`` only support a limited number of types:

* Numerical values: int, float, double, ...
* string
* Function (in the form of PackedFunc)
* NDArray

Though limited, the above type system is very powerful because it supports
function as a first-class citizen. For example, if you want to return multiple
values, you can return a PackedFunc which returns each value given an integer
index. However, in many cases, new types are still desired to ease the
development process:

* The argument/return value is a composition of collections (e.g. dictionary of
  dictionary of list).
* Sometimes we just want to have a notion of "structure" (e.g. given an apple,
  get its color by ``apple.color``).

To achieve this, we introduce the Object type system. For example, to define a
new type ``Calculator``:

.. code:: c++

   // file: calculator.cc
   #include <dgl/packed_func_ext.h>
   using namespace runtime;
   class CalculatorObject : public Object {
    public:
     std::string brand;
     int price;
     
     void VisitAttrs(AttrVisitor *v) final {
       v->Visit("brand", &brand);
       v->Visit("price", &price);
     }

     static constexpr const char* _type_key = "Calculator";
     DGL_DECLARE_OBJECT_TYPE_INFO(CalculatorObject, Object);
   };

   // This is to define a reference class (the wrapper of an object shared pointer).
   // A minimal implementation is as follows, but you could define extra methods.
   class Calculator : public ObjectRef {
    public:
     const CalculatorObject* operator->() const {
       return static_cast<const CalculatorObject*>(obj_.get());
     }
     using ContainerType = CalculatorObject;
   };

   DGL_REGISTER_GLOBAL("calculator.CreateCaculator")
   .set_body([] (DGLArgs args, DGLRetValue* rv) {
     std::string brand = args[0];
     int price = args[1];
     auto o = std::make_shared<CalculatorObject>();
     o->brand = brand;
     o->price = price;
     *rv = o;
   }

On the python side:

.. code:: python

   # file: calculator.py
   from dgl._ffi.object import register_object, ObjectBase
   from ._ffi.function import _init_api

   @register_object
   class Calculator(ObjectBase):
     @staticmethod
     def create(brand, price):
       # invoke a C API, the return value is of `Calculator` type
       return CreateCalculator(brand, price)

   _init_api("dgl.calculator")

We can then simply create ``Calculator`` object by:

.. code:: python

   calc = Calculator.create("casio", 100)

What is nice about this object is that, it defines a visitor pattern that is
essentially a reflection mechanism to get its internal attributes. For example,
you can print the calculator's brand and by simply accessing its attributes.

.. code:: python

   print(calc.brand)
   print(calc.price)

The reflection is indeed a little bit slow due to the string key lookup. To
speed it up, you could define an attribute access API:

.. code:: c++

   // file: calculator.cc
   DGL_REGISTER_GLOBAL("calculator.CaculatorGetBrand")
   .set_body([] (DGLArgs args, DGLRetValue* rv) {
     Calculator calc = args[0];
     *rv = calc->brand;
   }

Containers
----------

Containers are also objects. For example, the C API below accepts a list of
integers and return their sum:

.. code:: c++

   // in file: calculator.cc
   #include <dgl/runtime/container.h>
   using namespace runtime;
   DGL_REGISTER_GLOBAL("calculator.Sum")
   .set_body([] (DGLArgs args, DGLRetValue* rv) {
     // All the DGL supported values are represented as a ValueObject, which
     //   contains a data field.
     List<Value> values = args[0];
     int sum = 0;
     for (int i = 0; i < values.size(); ++i) {
       sum += static_cast<int>(values[i]->data);
     }
   }

Invoking this API is simple -- just pass a python list of integers. DGL FFI will
automatically convert python list/tuple/dictionary to the corresponding object
type.

.. code:: python

   # in file: calculator.py
   from ._ffi.function import _init_api

   Sum([0, 1, 2, 3, 4, 5])

   _init_api("dgl.calculator")

The elements in the containers can be any objects, which allows the containers
to be composed. Below is an API that accepts a list of calculators and print
out their price:

.. code:: c++

   // in file: calculator.cc
   #include <iostream>
   #include <dgl/runtime/container.h>
   using namespace runtime;
   DGL_REGISTER_GLOBAL("calculator.PrintCalculators")
   .set_body([] (DGLArgs args, DGLRetValue* rv) {
     List<Calculator> calcs = args[0];
     for (int i = 0; i < calcs.size(); ++i) {
       std::cout << calcs[i]->price << std::endl;
     }
   }

Please note that containers are NOT meant for passing a large collection of
items from/to C APIs. It will be quite slow in these cases. It is recommended
to benchmark first. As an alternative, use NDArray for a large collection of
numerical values and use ``dgl.batch`` to batch a lot of ``DGLGraph``'s into 
a single ``DGLGraph``.
