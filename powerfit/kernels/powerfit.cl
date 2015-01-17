
#include <pyopencl-complex.h>

//__kernel
//void c_conj_multiply(__global cfloat_t *x, __global cfloat_t *y, 
//        __global cfloat_t *out)
//{
//    /* Calculate complex conjugate of x and multiply with y
//     * x can be multiple times the size of y
//     */
//
//    int ind, offset, i, j;
//    int id = get_global_id(0);
//    int stride = get_global_size(0);
//
//    int xsize = sizeof(x);
//    int ysize = sizeof(y);
//
//    for (j = 0; j < repeats; j++){
//
//        offset = j*size;
//
//        for (i = id; i < size; i += stride){
//            ind = i + offset;
//            out[ind] = cfloat_mul(cfloat_conj(x[ind], y[i]));
//        }
//    }
//}

__kernel
void r_multiply(__global float *x, __global float *y, 
        __global float *out)
{
    /* Calculate complex conjugate of x and multiply with y
     * x can be multiple times the size of y
     */

    int id = get_global_id(0);
    int stride = get_global_size(0);
    int repeats = sizeof x/ sizeof y;
    int yside = sizeof y/ sizeof *y;
    int offset, ind;

    for (j = 0; j < repeats; j++){

        offset = j*ysize;

        for (i = id; i < ysize; i += stride){
            ind = i + offset;
            out[ind] = x[ind] * y[i];
        }
    }
}
