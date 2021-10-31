#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "stats.h"

static char module_docstring[] = "GSL Statistics module";
static char t_test_docstring[] = "Perform an independent samples t-test";
static char ttest_all_docstring[] = "Perform independent samples t-test between every pair of groups";

static char trace_docstring[] = "Calculate the trace of the matrix";

static PyObject *trace(PyObject *self, PyObject *args)
{
    PyArrayObject *array;
    double sum;
    int i, n;
     
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
        return NULL;
    if (array->nd != 2 || array->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
        return NULL;
    }
     
    n = array->dimensions[0];
    if (n > array->dimensions[1])
        n = array->dimensions[1];

    sum = 0.;
    for (i = 0; i < n; i++)
        sum += *(double *)(array->data + i*array->strides[0] + i*array->strides[1]);
     
    return PyFloat_FromDouble(sum);
}

static PyObject *ttest_all(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject *array;
    
    if (!PyArg_ParseTuple(args, "O", &input))
        return NULL;
    array = (PyArrayObject *) PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 2, 2);
    if (array->nd != 2 || array->descr->type_num != PyArray_DOUBLE)
    {
        PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
        return NULL;
    }

    // Create output array
    int ngroups = array->dimensions[1];
    int odims[2] = {ngroups, ngroups};
    //PyObject *out = PyArray_FromDims(2, odims, PyArray_DOUBLE);

    // Now do t-tests between each pair of groups
    for (int i = 0; i < ngroups; i++) {
        // Get group i slice
        PyObject* slice_xi = PySlice_New(NULL, NULL, NULL);
        //PyObject* slice_yi = PySlice_New(PyLong_FromLong(0), PyLong_FromLong(array->dimensions[1]), NULL);
        PyObject* slice_yi = PyInt_FromLong(0);
        PyObject* slices_i = PyTuple_Pack(2, slice_xi, slice_yi);
        PyArrayObject* xdata = PyObject_GetItem(input, slices_i);
        double *x = (double *) xdata->data;
        for (int j = 0; j < array->dimensions[1]; j++) {
            printf("%lf\n", x[j]);
        }
        for (int j = i+1; j < ngroups; j++) {
            // Get next group slice
            PyObject* slice_xj = PySlice_New(NULL, NULL, NULL);
            PyObject* slice_yj = PySlice_New(PyLong_FromLong(0), PyLong_FromLong(array->dimensions[1]), NULL);
            PyObject* slices_j = PyTuple_Pack(2, slice_xj, slice_yj);
            PyArrayObject* ydata = PyObject_GetItem((PyObject *) array, slices_j);
            double *y = (double *) ydata->data;
        }
    }
    return Py_BuildValue("d", 1);
}

static PyObject *_py_ttest(PyObject *self, PyObject *args)
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
    res = ttest_equalvar(d1, n1, d2, n2);
    return Py_BuildValue("d", res);
}

static PyObject *
matrix_vector(PyObject *self, PyObject *args)
{
    PyObject *input1, *input2;
    PyArrayObject *matrix, *vector, *result;
    int dimensions[1];
    double factor[1];
    double real_zero[1] = {0.};
    long int_one[1] = {1};
    long dim0[1], dim1[1];
     
    extern dgemv_(char *trans, long *m, long *n,
        double *alpha, double *a, long *lda,
        double *x, long *incx,
        double *beta, double *Y, long *incy);
     
    if (!PyArg_ParseTuple(args, "dOO", factor, &input1, &input2))
        return NULL;
    matrix = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 2, 2);
    if (matrix == NULL)
        return NULL;
    vector = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_DOUBLE, 1, 1);
    if (vector == NULL)
        return NULL;
    if (matrix->dimensions[1] != vector->dimensions[0]) {
        PyErr_SetString(PyExc_ValueError, "array dimensions are not compatible");
        return NULL;
    }
     
    dimensions[0] = matrix->dimensions[0];
    result = (PyArrayObject *)PyArray_FromDims(1, dimensions, PyArray_DOUBLE);
    if (result == NULL)
        return NULL;
     
    dim0[0] = (long)matrix->dimensions[0];
    dim1[0] = (long)matrix->dimensions[1];
    dgemv_("T", dim1, dim0, factor, (double *)matrix->data, dim1,
        (double *)vector->data, int_one,
        real_zero, (double *)result->data, int_one);
     
    return PyArray_Return(result);
}

static PyMethodDef module_methods[] = {
    {"ttest", _py_ttest, METH_VARARGS, t_test_docstring},
    {"ttest_all", ttest_all, METH_VARARGS, ttest_all_docstring},
    {"trace", trace, METH_VARARGS, trace_docstring},
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
