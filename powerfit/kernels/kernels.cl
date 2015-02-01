__kernel
void rotate_image3d(sampler_t sampler,
                    read_only image3d_t image,
                    float16 rotmat, 
                   __global float *out, 
                   float4 center, 
                   int4 shape)
{
    int id = get_global_id(0);
    int stride = get_global_size(0);

    int x, y, z, slice;
    float xrot, yrot, zrot;
    float4 coordinate, weight;

    slice = shape.s2*shape.s1;

    int i;
    for (i = id; i < shape.s3; i += stride) {

        z = i/slice;
        y = (i - z*slice)/shape.s2;
        x = i - z*slice - y*shape.s2;

        if (x >= 0.5*shape.s2)
            x -= shape.s2;
        if (y >= 0.5*shape.s1)
            y -= shape.s1;
        if (z >= 0.5*shape.s0)
            z -= shape.s0;

        xrot = rotmat.s0*x + rotmat.s1*y + rotmat.s2*z;
        yrot = rotmat.s3*x + rotmat.s4*y + rotmat.s5*z;
        zrot = rotmat.s6*x + rotmat.s7*y + rotmat.s8*z;

        xrot += 0.5f + center.s0;
        yrot += 0.5f + center.s1;
        zrot += 0.5f + center.s2;
    
        coordinate = (float4) (xrot, yrot, zrot, 0);
        weight = read_imagef(image, sampler, coordinate);

        out[i] = weight.s0;
    }
}
__kernel
void count(__global int *interspace,
           __global int *access_interspace,
           float weight,
           __global float *counts,
           int size)
{
    int id = get_global_id(0);
    int stride = get_global_size(0);
    int i, n;

    for (i = id; i < size; i += stride){
        if (interspace[i] > 0)
            counts[i] += weight;

        n = access_interspace[i];
        if (n > 0)
            counts[n*size + i] += weight;
    }
}

__kernel
void blur_points(__global float4 *points,
                 __global float *weights,
                 float sigma,
                 __global float *out,
                 int4 shape, int npoints)
{
    const int zid = get_global_id(0);
    const int yid = get_global_id(1);
    const int xid = get_global_id(2);

    const int zstride = get_global_size(0);
    const int ystride = get_global_size(1);
    const int xstride = get_global_size(2);

    const float sigma2 = pown(sigma, 2);
    const float extend2 = pown(4*sigma, 2);
    const int slice = shape.s2 * shape.s1;
    const float hx = 0.5 * shape.s2;
    const float hy = 0.5 * shape.s1;
    const float hz = 0.5 * shape.s0;

    unsigned int i, iz, iy, ix, z_ind, yz_ind;
    float z, y, x, z2, y2z2, x2y2z2;

    for (i = 0; i < npoints; i ++){

        for (iz = zid; iz < shape.s0; iz += zstride){

            z = iz;
            if (z > hz)
                z -= shape.s0;
                
            z2 = pown(z - points[i].s0, 2);

            if (z2 > extend2)
                continue;

            z_ind = iz*slice;

            for (iy = yid; iy < shape.s1; iy += ystride){
                y = iy;
                if (y > hy)
                    y -= shape.s1;
                    
                y2z2 = pown(y - points[i].s1, 2) + z2;

                if (y2z2 > extend2)
                    continue;

                yz_ind = iy*shape.s2 + z_ind;

                for (ix = xid; ix < shape.s2; ix += xstride){
                    x = ix;
                    if (x > hx)
                        x -= shape.s2;
                        
                    x2y2z2 = pown(x - points[i].s2, 2) + y2z2;

                    if (x2y2z2 > extend2)
                        continue;

                    out[yz_ind + ix] += weights[i]*exp(-x2y2z2/sigma2);
                }
            }
        }
    }
}

__kernel
void dilate_points_add(__global float8 *constraints, 
                       float16 rotmat, 
                       __global int *restspace, int4 shape, int nrestraints)
{

    int zid = get_global_id(0);
    int yid = get_global_id(1);
    int xid = get_global_id(2);
    
    int zstride = get_global_size(0);
    int ystride = get_global_size(1);
    int xstride = get_global_size(2);

    int i, ix, iy, iz;
    int z_ind, yz_ind, xyz_ind, slice;
    float xligand, yligand, zligand;
    float xcenter, ycenter, zcenter, distance2, z_dis2, yz_dis2, xyz_dis2;

    slice = shape.s2 * shape.s1;

    for (i = 0; i < nrestraints; i++){

         // determine the center of the point that will be dilated
         xligand = rotmat.s0 * constraints[i].s3 + rotmat.s1 * constraints[i].s4 + rotmat.s2 * constraints[i].s5;
         yligand = rotmat.s3 * constraints[i].s3 + rotmat.s4 * constraints[i].s4 + rotmat.s5 * constraints[i].s5;
         zligand = rotmat.s6 * constraints[i].s3 + rotmat.s7 * constraints[i].s4 + rotmat.s8 * constraints[i].s5;

         xcenter = constraints[i].s0 - xligand;
         ycenter = constraints[i].s1 - yligand;
         zcenter = constraints[i].s2 - zligand;

         distance2 = pown(constraints[i].s6, 2);

         // calculate the distance of every voxel to the determined center
         for (iz = zid; iz < shape.s0; iz += zstride){

             z_dis2 = pown(iz - zcenter, 2);

             z_ind = iz * slice;

             for (iy = yid; iy < shape.s1; iy += ystride){
                 yz_dis2 = pown(iy - ycenter, 2) + z_dis2;

                 yz_ind = z_ind + iy*shape.s2;

                 for (ix = xid; ix < shape.s2; ix += xstride){

                     xyz_dis2 = pown(ix - xcenter, 2) + yz_dis2;

                     if (xyz_dis2 <= distance2){
                         xyz_ind = ix + yz_ind;
                         restspace[xyz_ind] += 1;
                     }
                 }
             }
         }
    }
}
