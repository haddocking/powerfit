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
