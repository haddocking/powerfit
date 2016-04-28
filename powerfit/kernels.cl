__kernel
void rotate_grids_and_multiply(
        read_only image3d_t template, read_only image3d_t mask, 
        global float* rotmat, sampler_t s_linear, sampler_t s_nearest, 
        float4 center, int4 shape, int radius, 
        global float* rot_template, global float* rot_mask,
        global float* rot_mask2, int nrot
        )
{
    /*Rotate the template and mask grid, and also calculate the mask2 grid
     *
     * Parameters
     * ----------
     * template
     *
     * mask
     *
     * rotmat
     *     Array that holds all the rotations.
     *
     * s_linear : sampler_t
     *     Sampler with LINEAR property.
     *
     * s_nearest : sampler_t
     *     Sampler with NEAREST property.
     *
     * center : float4
     *     Center around which the images are rotated.
     *
     *
     * shape : int4
     *    Contains the shape of output arrays, with the fourth element the size.
     *
     * radius : int
     *    Largest radius of image from center. All voxels within this radius
     *    will be rotated
     *
     * nrot : uint
     *     Index of the initial rotation that is sampled.
     *
     * Notes
     * -----
     */

    /* there is an offset of a half when sampling images properly */
    const float OFFSET = 0.5f;
    int radius2 = radius * radius;

    int slice, rotmat_offset;
    float4 weight, dist2;
    float4 coor, coor_z, coor_zy, coor_zyx;
    int4 index;
    int z, y, x;

    size_t zid = get_global_id(0);
    size_t yid = get_global_id(1);
    size_t xid = get_global_id(2);
    size_t zstride = get_global_size(0);
    size_t ystride = get_global_size(1);
    size_t xstride = get_global_size(2);

    /* Some precalculations */
    slice = shape.s2 * shape.s1;
    coor_zyx.s3 = 0;

    rotmat_offset = nrot * 9;
    coor.s0 = center.s0 + OFFSET;
    coor.s1 = center.s1 + OFFSET;
    coor.s2 = center.s2 + OFFSET;

    /* Loop over the grids */
    for (z = zid - radius; z <= radius; z += zstride) {
        dist2.s2 = z * z;
        coor_z.s0 = rotmat[rotmat_offset + 2] * z + coor.s0;
        coor_z.s1 = rotmat[rotmat_offset + 5] * z + coor.s1;
        coor_z.s2 = rotmat[rotmat_offset + 8] * z + coor.s2;

        index.s0 = z * slice;
        /* Wraparound the z-coordinate */
        if (z < 0)
            index.s0 += shape.s3;

        for (y = yid - radius; y <= radius; y += ystride) {
            dist2.s1 = y * y + dist2.s2;
            coor_zy.s0 = rotmat[rotmat_offset + 1] * y + coor_z.s0;
            coor_zy.s1 = rotmat[rotmat_offset + 4] * y + coor_z.s1;
            coor_zy.s2 = rotmat[rotmat_offset + 7] * y + coor_z.s2;

            index.s1 = index.s0 + y * shape.s2;
            /* Wraparound the y-coordinate */
            if (y < 0)
                index.s1 += slice;

            for (x = xid - radius; x <= radius; x += xstride) {
                dist2.s0 = x * x + dist2.s1;
                if (dist2.s0 > radius2)
                    continue;

                coor_zyx.s0 = rotmat[rotmat_offset + 0] * x + coor_zy.s0;
                coor_zyx.s1 = rotmat[rotmat_offset + 3] * x + coor_zy.s1;
                coor_zyx.s2 = rotmat[rotmat_offset + 6] * x + coor_zy.s2;

                index.s2 = index.s1 + x;
                if (x < 0)
                    index.s2 += shape.s2;

                weight = read_imagef(template, s_linear, coor_zyx);
                rot_template[index.s2] = weight.s0;
                weight = read_imagef(mask, s_nearest, coor_zyx);
                rot_mask[index.s2] = weight.s0;
                rot_mask2[index.s2] = weight.s0 * weight.s0;
            }
        }
    }
}
