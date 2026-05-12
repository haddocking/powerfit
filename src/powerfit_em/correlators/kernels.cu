#define IMAGE_OFFSET 0.5f

// To be defined on compile time
#define SHAPE_X $shape_x
#define SHAPE_Y $shape_y
#define SHAPE_Z $shape_z
#define LLENGTH $llength

#define INV_SHAPE_X (1.0f / (float)SHAPE_X)
#define INV_SHAPE_Y (1.0f / (float)SHAPE_Y)
#define INV_SHAPE_Z (1.0f / (float)SHAPE_Z)
#define LLENGTH2 (LLENGTH * LLENGTH)

__device__ inline int flat_index(int x, int y, int z) {
    return (z * SHAPE_Y + y) * SHAPE_X + x;
}

extern "C" __global__
void rotate_image3d_linear(cudaTextureObject_t tex, const float *rotmat, float *out) {
    int ox = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int oy = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    int oz = (int) (blockIdx.z * blockDim.z + threadIdx.z);

    if (ox >= SHAPE_X || oy >= SHAPE_Y || oz >= SHAPE_Z) {
        return;
    }

    int out_idx = flat_index(ox, oy, oz);

    int x;
    if (ox <= LLENGTH) {
        x = ox;
    } else if (ox >= SHAPE_X - LLENGTH) {
        x = ox - SHAPE_X;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int y;
    if (oy <= LLENGTH) {
        y = oy;
    } else if (oy >= SHAPE_Y - LLENGTH) {
        y = oy - SHAPE_Y;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int z;
    if (oz <= LLENGTH) {
        z = oz;
    } else if (oz >= SHAPE_Z - LLENGTH) {
        z = oz - SHAPE_Z;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int dist2 = x * x + y * y + z * z;
    if (dist2 > LLENGTH2) {
        out[out_idx] = 0.0f;
        return;
    }

    float sx = (rotmat[0] * (float)x + rotmat[3] * (float)y + rotmat[6] * (float)z + IMAGE_OFFSET) * INV_SHAPE_X;
    float sy = (rotmat[1] * (float)x + rotmat[4] * (float)y + rotmat[7] * (float)z + IMAGE_OFFSET) * INV_SHAPE_Y;
    float sz = (rotmat[2] * (float)x + rotmat[5] * (float)y + rotmat[8] * (float)z + IMAGE_OFFSET) * INV_SHAPE_Z;

    out[out_idx] = tex3D<float>(tex, sx, sy, sz);
}

extern "C" __global__
void rotate_image3d_linear_batch(cudaTextureObject_t tex, const float *rotmats, float *out, int n_batch) {
    int ox = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int oy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int tid_z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    int oz = tid_z % SHAPE_Z;
    int b  = tid_z / SHAPE_Z;

    if (ox >= SHAPE_X || oy >= SHAPE_Y || b >= n_batch) {
        return;
    }

    int out_idx = b * (SHAPE_Z * SHAPE_Y * SHAPE_X) + flat_index(ox, oy, oz);

    int x;
    if (ox <= LLENGTH) {
        x = ox;
    } else if (ox >= SHAPE_X - LLENGTH) {
        x = ox - SHAPE_X;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int y;
    if (oy <= LLENGTH) {
        y = oy;
    } else if (oy >= SHAPE_Y - LLENGTH) {
        y = oy - SHAPE_Y;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int z;
    if (oz <= LLENGTH) {
        z = oz;
    } else if (oz >= SHAPE_Z - LLENGTH) {
        z = oz - SHAPE_Z;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int dist2 = x * x + y * y + z * z;
    if (dist2 > LLENGTH2) {
        out[out_idx] = 0.0f;
        return;
    }

    const float *rotmat = rotmats + b * 9;
    float sx = (rotmat[0] * (float)x + rotmat[3] * (float)y + rotmat[6] * (float)z + IMAGE_OFFSET) * INV_SHAPE_X;
    float sy = (rotmat[1] * (float)x + rotmat[4] * (float)y + rotmat[7] * (float)z + IMAGE_OFFSET) * INV_SHAPE_Y;
    float sz = (rotmat[2] * (float)x + rotmat[5] * (float)y + rotmat[8] * (float)z + IMAGE_OFFSET) * INV_SHAPE_Z;

    out[out_idx] = tex3D<float>(tex, sx, sy, sz);
}

extern "C" __global__
void rotate_image3d_nearest_batch(cudaTextureObject_t tex, const float *rotmats, float *out, int n_batch) {
    int ox = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int oy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int tid_z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    int oz = tid_z % SHAPE_Z;
    int b  = tid_z / SHAPE_Z;

    if (ox >= SHAPE_X || oy >= SHAPE_Y || b >= n_batch) {
        return;
    }

    int out_idx = b * (SHAPE_Z * SHAPE_Y * SHAPE_X) + flat_index(ox, oy, oz);

    int x;
    if (ox <= LLENGTH) {
        x = ox;
    } else if (ox >= SHAPE_X - LLENGTH) {
        x = ox - SHAPE_X;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int y;
    if (oy <= LLENGTH) {
        y = oy;
    } else if (oy >= SHAPE_Y - LLENGTH) {
        y = oy - SHAPE_Y;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int z;
    if (oz <= LLENGTH) {
        z = oz;
    } else if (oz >= SHAPE_Z - LLENGTH) {
        z = oz - SHAPE_Z;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int dist2 = x * x + y * y + z * z;
    if (dist2 > LLENGTH2) {
        out[out_idx] = 0.0f;
        return;
    }

    const float *rotmat = rotmats + b * 9;
    float sx = (rotmat[0] * (float)x + rotmat[3] * (float)y + rotmat[6] * (float)z + IMAGE_OFFSET) * INV_SHAPE_X;
    float sy = (rotmat[1] * (float)x + rotmat[4] * (float)y + rotmat[7] * (float)z + IMAGE_OFFSET) * INV_SHAPE_Y;
    float sz = (rotmat[2] * (float)x + rotmat[5] * (float)y + rotmat[8] * (float)z + IMAGE_OFFSET) * INV_SHAPE_Z;

    out[out_idx] = tex3D<float>(tex, sx, sy, sz);
}

extern "C" __global__
void rotate_image3d_nearest(cudaTextureObject_t tex, const float *rotmat, float *out) {
    int ox = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    int oy = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    int oz = (int) (blockIdx.z * blockDim.z + threadIdx.z);

    if (ox >= SHAPE_X || oy >= SHAPE_Y || oz >= SHAPE_Z) {
        return;
    }

    int out_idx = flat_index(ox, oy, oz);

    int x;
    if (ox <= LLENGTH) {
        x = ox;
    } else if (ox >= SHAPE_X - LLENGTH) {
        x = ox - SHAPE_X;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int y;
    if (oy <= LLENGTH) {
        y = oy;
    } else if (oy >= SHAPE_Y - LLENGTH) {
        y = oy - SHAPE_Y;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int z;
    if (oz <= LLENGTH) {
        z = oz;
    } else if (oz >= SHAPE_Z - LLENGTH) {
        z = oz - SHAPE_Z;
    } else {
        out[out_idx] = 0.0f;
        return;
    }

    int dist2 = x * x + y * y + z * z;
    if (dist2 > LLENGTH2) {
        out[out_idx] = 0.0f;
        return;
    }

    float sx = (rotmat[0] * (float)x + rotmat[3] * (float)y + rotmat[6] * (float)z + IMAGE_OFFSET) * INV_SHAPE_X;
    float sy = (rotmat[1] * (float)x + rotmat[4] * (float)y + rotmat[7] * (float)z + IMAGE_OFFSET) * INV_SHAPE_Y;
    float sz = (rotmat[2] * (float)x + rotmat[5] * (float)y + rotmat[8] * (float)z + IMAGE_OFFSET) * INV_SHAPE_Z;

    out[out_idx] = tex3D<float>(tex, sx, sy, sz);
}

extern "C" __global__
void powerfit_batch_lcc_and_take_best(
    const float* __restrict__ gcc_batch,
    const float* __restrict__ ave_batch,
    const float* __restrict__ ave2_batch,
    const int*   __restrict__ lcc_mask,
    float* __restrict__ lcc,
    int*   __restrict__ rot,
    float norm_factor,
    int   batch_start,
    int   actual_batch,
    int   volume_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= volume_size) return;

    float best_lcc = lcc[i];
    int   best_rot = rot[i];
    const int mask = lcc_mask[i];

    if (mask != 0) {
        for (int b = 0; b < actual_batch; b++) {
            const int   idx     = b * volume_size + i;
            const float ave_val = ave_batch[idx];
            const float var     = ave2_batch[idx] * norm_factor - ave_val * ave_val;
            if (var > 0.0f) {
                const float score = gcc_batch[idx] / sqrtf(var);
                if (score > best_lcc) {
                    best_lcc = score;
                    best_rot = batch_start + b;
                }
            }
        }
    }

    lcc[i] = best_lcc;
    rot[i] = best_rot;
}