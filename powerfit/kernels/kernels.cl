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
void rotate_model_and_mask(sampler_t sampler,
                    read_only image3d_t im_modelmap,
                    read_only image3d_t im_mask,
                    float16 rotmat, 
                   __global float *modelmap, 
                   __global float *mask, 
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

        weight = read_imagef(im_modelmap, sampler, coordinate);
        modelmap[i] = weight.s0;

        weight = read_imagef(im_mask, sampler, coordinate);
        mask[i] = weight.s0;
    }
}
