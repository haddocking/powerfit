#define IMAGE_OFFSET 0.5f

// To be defined on compile time
#define SHAPE_X $shape_x
#define SHAPE_Y $shape_y
#define SHAPE_Z $shape_z
#define LLENGTH $llength

#define LLENGTH2 (LLENGTH * LLENGTH)

__device__ inline int wrap_index(int idx, int size) {
    int wrapped = idx % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

__device__ inline int flat_index(int x, int y, int z) {
    return (z * SHAPE_Y + y) * SHAPE_X + x;
}

__device__ inline float sample_nearest(const float *image, float x, float y, float z) {
    int ix = wrap_index((int) floorf(x), SHAPE_X);
    int iy = wrap_index((int) floorf(y), SHAPE_Y);
    int iz = wrap_index((int) floorf(z), SHAPE_Z);
    return image[flat_index(ix, iy, iz)];
}

__device__ inline float sample_linear(const float *image, float x, float y, float z) {
    float x_shift = x - 0.5f;
    float y_shift = y - 0.5f;
    float z_shift = z - 0.5f;

    int x0 = (int) floorf(x_shift);
    int y0 = (int) floorf(y_shift);
    int z0 = (int) floorf(z_shift);

    float fx = x_shift - (float) x0;
    float fy = y_shift - (float) y0;
    float fz = z_shift - (float) z0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    x0 = wrap_index(x0, SHAPE_X);
    y0 = wrap_index(y0, SHAPE_Y);
    z0 = wrap_index(z0, SHAPE_Z);
    x1 = wrap_index(x1, SHAPE_X);
    y1 = wrap_index(y1, SHAPE_Y);
    z1 = wrap_index(z1, SHAPE_Z);

    float c000 = image[flat_index(x0, y0, z0)];
    float c100 = image[flat_index(x1, y0, z0)];
    float c010 = image[flat_index(x0, y1, z0)];
    float c110 = image[flat_index(x1, y1, z0)];
    float c001 = image[flat_index(x0, y0, z1)];
    float c101 = image[flat_index(x1, y0, z1)];
    float c011 = image[flat_index(x0, y1, z1)];
    float c111 = image[flat_index(x1, y1, z1)];

    float c00 = c000 + fx * (c100 - c000);
    float c10 = c010 + fx * (c110 - c010);
    float c01 = c001 + fx * (c101 - c001);
    float c11 = c011 + fx * (c111 - c011);

    float c0 = c00 + fy * (c10 - c00);
    float c1 = c01 + fy * (c11 - c01);

    return c0 + fz * (c1 - c0);
}

extern "C" __global__
void rotate_image3d_linear(const float *image, const float *rotmat, float *out) {
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x) - LLENGTH;
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y) - LLENGTH;
    int z = (int) (blockIdx.z * blockDim.z + threadIdx.z) - LLENGTH;

    if (x > LLENGTH || y > LLENGTH || z > LLENGTH) {
        return;
    }

    int dist2 = x * x + y * y + z * z;
    if (dist2 > LLENGTH2) {
        return;
    }

    float sx = rotmat[0] * (float) x + rotmat[3] * (float) y + rotmat[6] * (float) z + IMAGE_OFFSET;
    float sy = rotmat[1] * (float) x + rotmat[4] * (float) y + rotmat[7] * (float) z + IMAGE_OFFSET;
    float sz = rotmat[2] * (float) x + rotmat[5] * (float) y + rotmat[8] * (float) z + IMAGE_OFFSET;

    int ox = wrap_index(x, SHAPE_X);
    int oy = wrap_index(y, SHAPE_Y);
    int oz = wrap_index(z, SHAPE_Z);

    out[flat_index(ox, oy, oz)] = sample_linear(image, sx, sy, sz);
}

extern "C" __global__
void rotate_image3d_nearest(const float *image, const float *rotmat, float *out) {
    int x = (int) (blockIdx.x * blockDim.x + threadIdx.x) - LLENGTH;
    int y = (int) (blockIdx.y * blockDim.y + threadIdx.y) - LLENGTH;
    int z = (int) (blockIdx.z * blockDim.z + threadIdx.z) - LLENGTH;

    if (x > LLENGTH || y > LLENGTH || z > LLENGTH) {
        return;
    }

    int dist2 = x * x + y * y + z * z;
    if (dist2 > LLENGTH2) {
        return;
    }

    float sx = rotmat[0] * (float) x + rotmat[3] * (float) y + rotmat[6] * (float) z + IMAGE_OFFSET;
    float sy = rotmat[1] * (float) x + rotmat[4] * (float) y + rotmat[7] * (float) z + IMAGE_OFFSET;
    float sz = rotmat[2] * (float) x + rotmat[5] * (float) y + rotmat[8] * (float) z + IMAGE_OFFSET;

    int ox = wrap_index(x, SHAPE_X);
    int oy = wrap_index(y, SHAPE_Y);
    int oz = wrap_index(z, SHAPE_Z);

    out[flat_index(ox, oy, oz)] = sample_nearest(image, sx, sy, sz);
}