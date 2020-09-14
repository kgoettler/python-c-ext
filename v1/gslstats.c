#include <stdio.h>
#include <math.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <Python.h>

static char module_docstring[] = "GSL Statistics module";
static char t_test_docstring[] = "Perform an independent samples t-test";

double t_test(const double *data1, int n1, const double *data2, int n2)
{
    double mean1, mean2, var1, var2, num, sp, denom, t, dof, p;
    mean1 = gsl_stats_mean(data1, 1, n1);
    mean2 = gsl_stats_mean(data2, 1, n2);
    var1 = gsl_stats_variance(data1, 1, n1);
    var2 = gsl_stats_variance(data2, 1, n2);
    num = mean1 - mean2;
    sp = sqrt((((n1 - 1) * var1) + ((n2 - 1) * var2)) / (n1 + n2 - 2));
    denom = sp * sqrt((1 / (double) n1) + (1 / (double) n2));
    t = fabs(num / denom);
    dof = (double) n1 + n2 - 2;
    p = (1 - gsl_cdf_tdist_P(t, dof)) * 2;
    return p;
}

static double* PySequence_ToDoubleArray(PyObject* inp, int n)
{
    // Allocate array
    double *out;
    out = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        PyObject *fitem;
        PyObject *item = PySequence_Fast_GET_ITEM(inp, i);
        if(!item) {
            Py_DECREF(inp);
            free(out);
            return NULL;
        }
        fitem = PyNumber_Float(item);
        if(!fitem) {
            Py_DECREF(inp);
            free(out);
            PyErr_SetString(PyExc_TypeError, "all items must be numbers");
            return NULL;
        }
        out[i] = PyFloat_AS_DOUBLE(fitem);
        Py_DECREF(fitem);
    }
    return out;
}

static PyObject *t_test_py(PyObject *self, PyObject *args)
{
    PyObject *inp1, *inp2;
    PyObject* data1;
    PyObject* data2;
    double *d1, *d2;
    int n1, n2;
    double res;

    if (!PyArg_ParseTuple(args, "OO", &inp1, &inp2))
        return 0;
    data1 = PySequence_Fast(inp1, "argument must be iterable");
    data2 = PySequence_Fast(inp2, "argument must be iterable");
    n1 = PySequence_Fast_GET_SIZE(data1);
    n2 = PySequence_Fast_GET_SIZE(data2);

    d1 = PySequence_ToDoubleArray(data1, n1);
    d2 = PySequence_ToDoubleArray(data2, n2);

    Py_DECREF(data1);
    Py_DECREF(data2);
    
    res = t_test(d1, n1, d2, n2);
    free(d1);
    free(d2);

    return Py_BuildValue("d", res);
}

static PyMethodDef module_methods[] = {
    {"t_test_py", t_test_py, METH_VARARGS, t_test_docstring},
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
    Py_Initialize();
    return PyModule_Create(&gslstatsmodule);
}
