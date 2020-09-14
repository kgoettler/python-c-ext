#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "ttest.h"

static char module_docstring[] = "GSL Statistics module";
static char t_test_docstring[] = "Perform an independent samples t-test";

static PyObject *ttest(PyObject *self, PyObject *args)
{
    PyArrayObject *data1, *data2;
    double *d1, *d2;
    int n1, n2;
    double res;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &data1, &PyArray_Type, &data2))
        return NULL;
    d1 = (double *) data1->data;
    d2 = (double *) data2->data;
    n1 = data1->dimensions[0];
    n2 = data2->dimensions[0];
    res = t_test(d1, n1, d2, n2);
    return Py_BuildValue("d", res);
}

static PyMethodDef module_methods[] = {
    {"ttest", ttest, METH_VARARGS, t_test_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gslstatsmodule = {
    PyModuleDef_HEAD_INIT,
    "gslstats",
    module_docstring,
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit_gslstats(void)
{
    import_array();
    return PyModule_Create(&gslstatsmodule);
}
