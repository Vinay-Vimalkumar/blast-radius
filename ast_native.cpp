#include <Python.h>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

namespace {

struct Decref {
    void operator()(PyObject* obj) const { Py_DECREF(obj); }
};
using Ref = std::unique_ptr<PyObject, Decref>;

Ref fields_for(PyTypeObject* type, PyObject* name) {
    PyObject* fields = PyDict_GetItemWithError(PyType_GetDict(type), name);
    if (!fields || !PyTuple_CheckExact(fields)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "unsupported AST fields");
        return {};
    }
    return Ref(Py_NewRef(fields));
}

int equal_tree(PyObject* left, PyObject* right, PyTypeObject* ast, PyObject* key, unsigned depth) {
    if (left == right) return 1;
    PyTypeObject* type = Py_TYPE(left);
    if (type != Py_TYPE(right) || depth == 256) return 0;
    if (type == &PyList_Type) {
        const Py_ssize_t size = PyList_GET_SIZE(left);
        if (size != PyList_GET_SIZE(right)) return 0;
        for (Py_ssize_t i = 0; i < size; ++i) {
            Ref a(PyList_GetItemRef(left, i)), b(PyList_GetItemRef(right, i));
            if (!a || !b) return -1;
            int equal = equal_tree(a.get(), b.get(), ast, key, depth + 1);
            if (equal != 1) return equal;
        }
        return 1;
    }
    if (!PyType_IsSubtype(type, ast)) return PyObject_RichCompareBool(left, right, Py_EQ);
    Ref fields = fields_for(type, key);
    if (!fields) return -1;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(fields.get()); ++i) {
        PyObject* field = PyTuple_GET_ITEM(fields.get(), i);
        if (!PyUnicode_Check(field)) {
            PyErr_SetString(PyExc_ValueError, "unsupported AST field name");
            return -1;
        }
        Ref a(PyObject_GetAttr(left, field));
        if (!a) return -1;
        Ref b(PyObject_GetAttr(right, field));
        if (!b) return -1;
        int equal = equal_tree(a.get(), b.get(), ast, key, depth + 1);
        if (equal != 1) return equal;
    }
    return 1;
}

bool append_text(std::string& out, PyObject* value) {
    Py_ssize_t size;
    const char* text = PyUnicode_AsUTF8AndSize(value, &size);
    if (!text) return false;
    out.append(text, static_cast<std::size_t>(size));
    return true;
}

bool append_node(std::string& out, PyObject* node, PyTypeObject* ast, PyObject* key, unsigned depth) {
    if (depth == 256) {
        PyErr_SetString(PyExc_RecursionError, "AST depth exceeded");
        return false;
    }
    PyTypeObject* type = Py_TYPE(node);
    if (type == &PyList_Type) {
        out += '[';
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(node); ++i) {
            if (i) out += ", ";
            Ref item(PyList_GetItemRef(node, i));
            if (!item || !append_node(out, item.get(), ast, key, depth + 1)) return false;
        }
        out += ']';
        return true;
    }
    if (!PyType_IsSubtype(type, ast)) {
        Ref repr(PyObject_Repr(node));
        return repr && append_text(out, repr.get());
    }
    Ref fields = fields_for(type, key);
    if (!fields) return false;
    const char* dot = std::strrchr(type->tp_name, '.');
    out += dot ? dot + 1 : type->tp_name;
    out += '(';
    bool first = true;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(fields.get()); ++i) {
        PyObject* field = PyTuple_GET_ITEM(fields.get(), i);
        if (!PyUnicode_Check(field)) {
            PyErr_SetString(PyExc_ValueError, "unsupported AST field name");
            return false;
        }
        Ref value(PyObject_GetAttr(node, field));
        if (!value) return false;
        if (PyList_CheckExact(value.get()) && PyList_GET_SIZE(value.get()) == 0) continue;
        if (value.get() == Py_None && PyDict_GetItem(PyType_GetDict(type), field) == Py_None) continue;
        if (!first) out += ", ";
        first = false;
        if (!append_text(out, field)) return false;
        out += '=';
        if (!append_node(out, value.get(), ast, key, depth + 1)) return false;
    }
    out += ')';
    return true;
}

PyObject* equal(PyObject*, PyObject* const* args, Py_ssize_t n) {
    if (n != 4 || !PyType_Check(args[2]) || !PyUnicode_CheckExact(args[3])) {
        PyErr_SetString(PyExc_TypeError, "expected two nodes, AST base type and field name");
        return nullptr;
    }
    int result = equal_tree(args[0], args[1], reinterpret_cast<PyTypeObject*>(args[2]), args[3], 0);
    return result < 0 ? nullptr : PyBool_FromLong(result);
}

PyObject* dump(PyObject*, PyObject* const* args, Py_ssize_t n) {
    if (n != 3 || !PyType_Check(args[1]) || !PyUnicode_CheckExact(args[2])) {
        PyErr_SetString(PyExc_TypeError, "expected node, AST base type and field name");
        return nullptr;
    }
    try {
        std::string out;
        out.reserve(256);
        if (!append_node(out, args[0], reinterpret_cast<PyTypeObject*>(args[1]), args[2], 0)) return nullptr;
        return PyUnicode_DecodeUTF8(out.data(), static_cast<Py_ssize_t>(out.size()), "strict");
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::length_error&) {
        PyErr_SetString(PyExc_OverflowError, "AST string too long");
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"equal", reinterpret_cast<PyCFunction>(equal), METH_FASTCALL, nullptr},
    {"dump", reinterpret_cast<PyCFunction>(dump), METH_FASTCALL, nullptr},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_ast_native", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__ast_native() { return PyModule_Create(&module); }
