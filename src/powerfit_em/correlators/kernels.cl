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


// Helper function: rotates all voxels within LLENGTH using the provided rotation matrix
// Encapsulates the triple-nested loop logic shared by rotate_image3d and rotate_image3d_batch
void rotate_voxel_loop(
        read_only image3d_t image,
        sampler_t sampler,
        float16 rotmat,
        float4 fshape,
        int out_base,
        int zid, int yid, int xid,
        int zstride, int ystride, int xstride,
        global float *out
        )
{
    int z, y, x;
    float4 dist2, coor_z, coor_zy, coor_zyx;
    int4 out_ind;

    for (z = zid - LLENGTH; z <= LLENGTH; z += zstride) {
        dist2.s2 = SQUARE(z);

        coor_z.s0 = rotmat.s6 * z + IMAGE_OFFSET;
        coor_z.s1 = rotmat.s7 * z + IMAGE_OFFSET;
        coor_z.s2 = rotmat.s8 * z + IMAGE_OFFSET;

        out_ind.s0 = out_base + z * SLICE;
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

    float4 fshape;
    fshape.s2 = (float) SHAPE_X;
    fshape.s1 = (float) SHAPE_Y;
    fshape.s0 = (float) SHAPE_Z;

    rotate_voxel_loop(image, sampler, rotmat, fshape, 0, zid, yid, xid, zstride, ystride, xstride, out);
}


kernel
void rotate_image3d_batch(
        read_only image3d_t image,
        sampler_t sampler,
        global const float *rotmats,
        int rot_offset,
        int batch_size,
        global float *out
        )
{
    // Unpack 3D work-item IDs: batch index packed into dimension 0 (Z iteration).
    // This allows full 3D parallelization over volume while processing
    // multiple rotations in parallel, matching the CUDA batch kernel strategy.
    int tid_z = get_global_id(0);  // [0, gws_z_single * batch_size)
    int yid = get_global_id(1);
    int xid = get_global_id(2);
    
    // Unpack batch and local Z iteration index
    int gws_z_single = get_global_size(0) / batch_size;  // Original 96 per batch
    int zid = tid_z % gws_z_single;
    int b = tid_z / gws_z_single;
    
    if (b >= batch_size)
        return;
    
    int zstride = get_global_size(0) / batch_size;
    int ystride = get_global_size(1);
    int xstride = get_global_size(2);
    
    // Fetch rotation matrix for this batch slot
    int rbase = (rot_offset + b) * 16;
    float16 rotmat = (float16)(
        rotmats[rbase + 0], rotmats[rbase + 1], rotmats[rbase + 2], rotmats[rbase + 3],
        rotmats[rbase + 4], rotmats[rbase + 5], rotmats[rbase + 6], rotmats[rbase + 7],
        rotmats[rbase + 8], rotmats[rbase + 9], rotmats[rbase + 10], rotmats[rbase + 11],
        rotmats[rbase + 12], rotmats[rbase + 13], rotmats[rbase + 14], rotmats[rbase + 15]
    );
    
    float4 fshape;
    int out_base = b * SIZE;  // Batch slot offset in output buffer
    fshape.s2 = (float) SHAPE_X;
    fshape.s1 = (float) SHAPE_Y;
    fshape.s0 = (float) SHAPE_Z;
    
    rotate_voxel_loop(image, sampler, rotmat, fshape, out_base, zid, yid, xid, zstride, ystride, xstride, out);
}


kernel
void powerfit_batch_lcc_and_take_best(
        global const float *gcc,
        global const float *ave,
        global const float *ave2,
        global const int *mask,
        global float *lcc,
        global int *grot,
        float norm_factor,
        int batch_start,
        int batch_size,
        int volume_size
        )
{
    int i = get_global_id(0);
    if (i >= volume_size)
        return;

    if (mask[i] == 0)
        return;

    float best_lcc = lcc[i];
    int best_rot = grot[i];

    for (int b = 0; b < batch_size; ++b) {
        int idx = b * volume_size + i;
        float var = ave2[idx] * norm_factor - ave[idx] * ave[idx];
        if (var > 0.0f) {
            float score = gcc[idx] / sqrt(var);
            if (score > best_lcc) {
                best_lcc = score;
                best_rot = batch_start + b;
            }
        }
    }

    lcc[i] = best_lcc;
    grot[i] = best_rot;
}


/*
 * Batch conjugate multiply: out[b][i] = conj(a[b][i]) * broadcast_b[i]
 *
 * a           : batch_size * ft_vol_size complex64 (float2) elements, batch-major
 * broadcast_b : ft_vol_size complex64 (float2) elements (same for every batch slot)
 * out         : batch_size * ft_vol_size complex64 (float2) output
 * ft_vol_size : number of complex elements per batch slot
 * total_size  : batch_size * ft_vol_size (guard against padded global work size)
 */
kernel
void powerfit_batch_conj_multiply(
        global const float2 *a,
        global const float2 *broadcast_b,
        global float2 *out,
        int ft_vol_size,
        int total_size
        )
{
    int i = get_global_id(0);
    if (i >= total_size)
        return;

    float2 av = a[i];
    float2 bv = broadcast_b[i % ft_vol_size];
    /* conj(av) * bv */
    out[i] = (float2)(av.x * bv.x + av.y * bv.y,
                      av.x * bv.y - av.y * bv.x);
}
