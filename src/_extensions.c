#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <complex.h>


static PyObject *rotate_grid3d(PyObject *dummy, PyObject *args)
{
    // Argument variables
    PyObject *arg1=NULL, *arg2=NULL, *arg4=NULL;
    int radius, nearest;
    PyArrayObject *py_grid=NULL, *py_rotmat=NULL, *py_out=NULL;

    // Pointers
    double *grid, *rotmat, *out;
    npy_intp *out_shape, *grid_shape;

    npy_intp out_size, out_slice, grid_size, grid_slice;
    int z, y, x, radius2, x0, y0, z0, dist2_z, dist2_zy, dist2_zyx, 
        out_z, out_zy, out_zyx, grid_zyx;
    double xcoor_z, ycoor_z, zcoor_z, xcoor_zy, ycoor_zy, zcoor_zy,
           xcoor_zyx, ycoor_zyx, zcoor_zyx;
    // Trilinear interpolation
    int x1, y1, z1, offset0, offset1;
    double dx, dy, dz, dx1, dy1, dz1, c00, c10, c01, c11, c0, c1, c;

    // Parse arguments
    if (!PyArg_ParseTuple(args, "OOiOi", &arg1, &arg2, &radius, &arg4, &nearest))
        return NULL;

    py_grid = (PyArrayObject *) PyArray_FROM_OTF(arg1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_grid == NULL)
        goto fail;
    py_rotmat = (PyArrayObject *) PyArray_FROM_OTF(arg2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (py_rotmat == NULL)
        goto fail;
    py_out = (PyArrayObject *) PyArray_FROM_OTF(arg4, NPY_FLOAT64, NPY_ARRAY_OUT_ARRAY);
    if (py_out == NULL)
        goto fail;

    // Get pointers to arrays and shape info
    grid = (double *) PyArray_DATA(py_grid);
    rotmat = (double *) PyArray_DATA(py_rotmat);
    out = (double *) PyArray_DATA(py_out);

    out_shape = PyArray_DIMS(py_out);
    out_slice = out_shape[2] * out_shape[1];
    out_size = PyArray_SIZE(py_out);
    grid_shape = PyArray_DIMS(py_grid);
    grid_slice = grid_shape[2] * grid_shape[1];
    grid_size = PyArray_SIZE(py_grid);

    radius2 = radius * radius;

    for (z = -radius; z <= radius; z++) {
        dist2_z = z * z;
        if (dist2_z > radius2)
            continue;
        // Rotate points
        xcoor_z = rotmat[6] * z;
        ycoor_z = rotmat[7] * z;
        zcoor_z = rotmat[8] * z;

        // Indices
        out_z = z * out_slice;
        if (z < 0)
            out_z += out_size;

        for (y = -radius; y <= radius; y++) {
            dist2_zy = dist2_z + y * y;
            if (dist2_zy > radius2)
                continue;

            xcoor_zy = xcoor_z + rotmat[3] * y;
            ycoor_zy = ycoor_z + rotmat[4] * y;
            zcoor_zy = zcoor_z + rotmat[5] * y;

            out_zy = out_z + y * out_shape[2];
            if (y < 0)
                out_zy += out_slice;

            for (x = -radius; x <= radius; x++) {
                dist2_zyx = dist2_zy + x * x;
                if (dist2_zyx > radius2)
                    continue;

                xcoor_zyx = xcoor_zy + rotmat[0] * x;
                ycoor_zyx = ycoor_zy + rotmat[1] * x;
                zcoor_zyx = zcoor_zy + rotmat[2] * x;
                // Indices
                out_zyx = out_zy + x;
                if (x < 0)
                    out_zyx += out_shape[2];

                if (nearest > 0) {
                    // Nearest interpolation
                    x0 = (int) round(xcoor_zyx);
                    y0 = (int) round(ycoor_zyx);
                    z0 = (int) round(zcoor_zyx);

                    // Grid index
                    grid_zyx = z0 * grid_slice + y0 * grid_shape[2] + x0;
                    if (x0 < 0)
                        grid_zyx += grid_shape[2];
                    if (y0 < 0)
                        grid_zyx += grid_slice;
                    if (z0 < 0)
                        grid_zyx += grid_size;

                    out[out_zyx] = grid[grid_zyx];
                } else {
                    // Tri-linear interpolation
                    //
                    x0 = (int) floor(xcoor_zyx);
                    y0 = (int) floor(ycoor_zyx);
                    z0 = (int) floor(zcoor_zyx);
                    x1 = x0 + 1;
                    y1 = y0 + 1;
                    z1 = z0 + 1;

                    // Grid index
                    grid_zyx = z0 * grid_slice + y0 * grid_shape[2] + x0;
                    if (x0 < 0)
                        grid_zyx += grid_shape[2];
                    if (y0 < 0)
                        grid_zyx += grid_slice;
                    if (z0 < 0)
                        grid_zyx += grid_size;

                    dx = xcoor_zyx - x0;
                    dy = ycoor_zyx - y0;
                    dz = zcoor_zyx - z0;
                    dx1 = 1 - dx;
                    dy1 = 1 - dy;
                    dz1 = 1 - dz;

                    offset1 = 1;
                    if (x1 == 0)
                        offset1 -= grid_shape[2];
                    c00 = grid[grid_zyx] * dx1 + 
                          grid[grid_zyx + offset1] * dx;

                    offset0 = grid_shape[2];
                    if (y1 == 0)
                        offset0 -= grid_slice;
                    offset1 = offset0 + 1;
                    if (x1 == 0)
                        offset1 -= grid_shape[2];

                    c10 = grid[grid_zyx + offset0] * dx1 + 
                          grid[grid_zyx + offset1] * dx;

                    offset0 = grid_slice;
                    if (z1 == 0)
                        offset0 -= grid_size;
                    offset1 = offset0 + 1;
                    if (x1 == 0)
                        offset1 -= grid_shape[2];
                    c01 = grid[grid_zyx + offset0] * dx1 + 
                          grid[grid_zyx + offset1] * dx;

                    offset0 = grid_slice + grid_shape[2];
                    if (z1 == 0)
                        offset0 -= grid_size;
                    if (y1 == 0)
                        offset0 -= grid_slice;
                    offset1 = offset0 + 1;
                    if (x1 == 0)
                        offset1 -= grid_shape[2];
                    c11 = grid[grid_zyx + offset0] * dx1 + 
                          grid[grid_zyx + offset1] * dx;

                    c0 = c00 * dy1 + c10 * dy;
                    c1 = c01 * dy1 + c11 * dy;

                    c = c0 * dz1 + c1 * dz;

                    out[out_zyx] = c;
                }
            }
        }
    }
    // Clean up objects
    Py_DECREF(py_grid);
    Py_DECREF(py_rotmat);
    Py_DECREF(py_out);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    // Clean up objects
    Py_XDECREF(py_grid);
    Py_XDECREF(py_rotmat);
    PyArray_XDECREF(py_out);
    return NULL;
}



static PyObject *conj_multiply(PyObject *dummy, PyObject *args)
{
    //Calculate conj(a) * (b)
    //

    PyObject *arg1=NULL, *arg2=NULL, *arg3=NULL;
    PyArrayObject *py_in1=NULL, *py_in2=NULL, *py_out=NULL;

    double complex *in1, *in2, *out;
    npy_intp out_size, i;

    // Parse arguments
    if (!PyArg_ParseTuple(args, "OOO", &arg1, &arg2, &arg3))
        return NULL;

    py_in1 = (PyArrayObject *) PyArray_FROM_OTF(arg1, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (py_in1 == NULL)
        goto fail;
    py_in2 = (PyArrayObject *) PyArray_FROM_OTF(arg2, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    if (py_in2 == NULL)
        goto fail;
    py_out = (PyArrayObject *) PyArray_FROM_OTF(arg3, NPY_COMPLEX128, NPY_ARRAY_OUT_ARRAY);
    if (py_out == NULL)
        goto fail;

    // Get pointers to arrays and shape info
    in1 = (double complex *) PyArray_DATA(py_in1);
    in2 = (double complex *) PyArray_DATA(py_in2);
    out = (double complex *) PyArray_DATA(py_out);
    out_size = PyArray_SIZE(py_out);

    for (i = 0; i < out_size; i++)
        out[i] = conj(in1[i]) * in2[i];

    // Clean up objects
    Py_DECREF(py_in1);
    Py_DECREF(py_in2);
    Py_DECREF(py_out);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    // Clean up objects
    Py_XDECREF(py_in1);
    Py_XDECREF(py_in2);
    PyArray_XDECREF(py_out);
    return NULL;
}


static PyMethodDef mymethods[] = {
    {"conj_multiply", conj_multiply, METH_VARARGS, ""},
    {"rotate_grid3d", rotate_grid3d, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


// NEW: Module definition struct for Python 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_extensions",        // name of module
    NULL,                 // docstring (or NULL)
    -1,                   // size of per-interpreter module state (-1 = global)
    mymethods             // your function table
};

// NEW: Module init function for Python 3
PyMODINIT_FUNC PyInit__extensions(void) {
    import_array();       // numpy setup
    return PyModule_Create(&moduledef);
}

// PyMODINIT_FUNC
// init_extensions(void)
// {
//     (void) Py_InitModule("_extensions", mymethods);
//     import_array();
// };
//
