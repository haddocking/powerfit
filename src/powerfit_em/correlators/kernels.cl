#define SQUARE(a) ((a) * (a))
#define IMAGE_OFFSET 0.5f

// To be defined on compile time
#define SHAPE_X $shape_x
#define SHAPE_Y $shape_y
#define SHAPE_Z $shape_z
#define LLENGTH $llength

#define LLENGTH2 (LLENGTH * LLENGTH)
#define SLICE ((SHAPE_X * SHAPE_Y))
#define SIZE ((SHAPE_Z * SLICE))


kernel
void rotate_image3d(
        read_only image3d_t image, sampler_t sampler, float16 rotmat, 
        global float *out
        )
{
    // Rotate grid around the origin. Only grid points within LLENGTH of the
    // origin are rotated.

    int zid = get_global_id(0);
    int yid = get_global_id(1);
    int xid = get_global_id(2);
    int zstride = get_global_size(0);
    int ystride = get_global_size(1);
    int xstride = get_global_size(2);

    int z, y, x;
    float4 dist2, coor_z, coor_zy, coor_zyx, fshape;
    int4 out_ind;
    fshape.s2 = (float) SHAPE_X;
    fshape.s1 = (float) SHAPE_Y;
    fshape.s0 = (float) SHAPE_Z;

    for (z = zid - LLENGTH; z <= LLENGTH; z += zstride) {
        dist2.s2 = SQUARE(z);

        coor_z.s0 = rotmat.s6 * z + IMAGE_OFFSET;
        coor_z.s1 = rotmat.s7 * z + IMAGE_OFFSET;
        coor_z.s2 = rotmat.s8 * z + IMAGE_OFFSET;

        out_ind.s0 = z * SLICE;
        if (z < 0)
            out_ind.s0 += SIZE;

        for (y = yid - LLENGTH; y <= LLENGTH; y += ystride) {
            dist2.s1 = SQUARE(y) + dist2.s2;
            if (dist2.s1 > LLENGTH2)
                continue;

            coor_zy.s0 = rotmat.s3 * y + coor_z.s0;
            coor_zy.s1 = rotmat.s4 * y + coor_z.s1;
            coor_zy.s2 = rotmat.s5 * y + coor_z.s2;

            out_ind.s1 = out_ind.s0 + y * SHAPE_X;
            if (y < 0)
                out_ind.s1 += SLICE;

            for (x = xid - LLENGTH; x <= LLENGTH; x += xstride) {
                dist2.s0 = SQUARE(x) + dist2.s1;
                if (dist2.s0 > LLENGTH2)
                    continue;
                // Normalize coordinates
                coor_zyx.s0 = (rotmat.s0 * x + coor_zy.s0) / fshape.s2;
                coor_zyx.s1 = (rotmat.s1 * x + coor_zy.s1) / fshape.s1;
                coor_zyx.s2 = (rotmat.s2 * x + coor_zy.s2) / fshape.s0;

                out_ind.s2 = out_ind.s1 + x;
                if (x < 0)
                    out_ind.s2 += SHAPE_X;

                out[out_ind.s2] = read_imagef(image, sampler, coor_zyx).s0;
            }
        }
    }
}
