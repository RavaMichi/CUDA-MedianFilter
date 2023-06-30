/****************************************************************************
 *
 * cuda-median-multi.cu - Image denoising using median filter
 *
 ****************************************************************************/

/***
% HPC - Image denoising using median filter
% Michele Ravaioli <michele.ravaioli3@unibo.it>
% Last modified: 2023-06-30

The file [cuda-median-multi.cu](cuda-median-multi.cu) contains a CUDA implementation
of an _image denoising_ algorithm which uses histograms to compute the median.
The algorithm replaces the color of each pixel with the _median_ of a neighborhood
of radius `RADIUS` (including itself). This codeuses multiple GPUs to perform
the computation faster.

To compile:

        nvcc cuda-median-multi.cu -o cuda-median-multi -Xcompiler -fopenmp

To execute:

        ./cuda-median-multi filein fileout width height radius

Input and output files are binary files representing matrices of
$\mathtt{width} \times \mathtt{height}$ elements of type `uint16_t`.
To show the image using [GIMP](https://www.gimp.org/), in the
File->Open dialog box check the "Select File Type" option at the
bottom and choose "Raw image data". Also, check "Show All Files" since
the file suffix is not recognized. Choose the file you want to open,
and in the "Load Image From Raw Data" dialog box select:

- Image Type: "Gray unsigned 16 bit Little Endian";
- Offset: 0;
- Width: set appropriately;
- Height: set appropriately;

## Files

- [cuda-median-multi.cu](cuda-median-multi.cu)
- [hpc.h](hpc.h)

 ***/

#include "hpc.h" /* MUST be the first file included */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <omp.h>

typedef uint16_t data_t;

// CUDA threads per CUDA block
#define BLKDIM 1024
// CUDA threads per group (= warp size)
#define TH_PER_GROUP 32
// n^ of groups per block
#define THREAD_GROUPS (BLKDIM / TH_PER_GROUP)

#define BASE_BITMASK 0xff00
#define VAL_BITMASK 0x00ff
#define HIST_SIZE 256

/*
 * Ghost Area Initialization Kernel
 */
__global__ void init_ghost_area( data_t* in, data_t* out, const int WIDTH, const int HEIGHT, const int RADIUS )
{
    const int GHOST_AREA_WIDTH = (2 * RADIUS) + WIDTH;
    /* Pixel coords */
    const int IDX = (threadIdx.x + (blockIdx.x * BLKDIM)) % GHOST_AREA_WIDTH;
    const int IDY = (threadIdx.x + (blockIdx.x * BLKDIM)) / GHOST_AREA_WIDTH;
    int x, y;

    if (IDX < RADIUS) {
        x = 0;
    } else if (IDX >= WIDTH + RADIUS) {
        x = WIDTH - 1;
    } else {
        x = IDX - RADIUS;
    }
    if (IDY < RADIUS) {
        y = 0;
    } else if (IDY >= HEIGHT + RADIUS) {
        y = HEIGHT - 1;
    } else {
        y = IDY - RADIUS;
    }

    /* Write pixel in output */
    out[IDX + (IDY * GHOST_AREA_WIDTH)] = in[x + (y * WIDTH)];
}

/*
 * Reinitialize histogram
 */
__device__ void init_hist( int* hist, int threadId )
{
    const int step = HIST_SIZE / TH_PER_GROUP;
#pragma unroll
    for (int i=threadId; i<HIST_SIZE; i+=step) {
        hist[i] = 0;
    }
    __syncwarp();
}

/*
 * Median Filtering using Histmedian algorithm
 */
__global__ void median_filter_kernel( data_t* in, data_t* out, const int WIDTH, const int HEIGHT, const int RADIUS, const int OFFSET )
{
    /* Thread IDs */
    const int th_group = threadIdx.x / TH_PER_GROUP;
    const int th_id = threadIdx.x % TH_PER_GROUP;

    /* Histogram in shared memory for performance boost */
    __shared__ int sh_hist[HIST_SIZE * THREAD_GROUPS];
    __shared__ data_t sh_base[THREAD_GROUPS];
    // Each thread group uses 256 elements of the shared array
    int *hist = &sh_hist[HIST_SIZE * th_group];

    /* Pixel coords */
    const int pixelX = (OFFSET + th_group + (blockIdx.x * THREAD_GROUPS)) % WIDTH;
    const int pixelY = (OFFSET + th_group + (blockIdx.x * THREAD_GROUPS)) / WIDTH;

    if (pixelY >= HEIGHT) return;
    
    const int WINDOW_L = (2 * RADIUS) + 1;
    const int WINDOW_SIZE = WINDOW_L * WINDOW_L;
    const int GHOST_AREA_WIDTH = (2 * RADIUS) + WIDTH;
    const int MEDIAN_POSITION = WINDOW_SIZE / 2;

    int count = 0; // used for median searching

    /*** Compute median ***/
    /* This proposed histogram median algorithm works only for elements of 16 bit */

    /* Init histogram */
    init_hist(hist, th_id);

    /* Compute histogram for greater 8 bits */
    for (int i=th_id; i<WINDOW_SIZE; i+=TH_PER_GROUP) {
        // window pixel coords
        const int win_pX = (i % WINDOW_L) + pixelX;
        const int win_pY = (i / WINDOW_L) + pixelY;

        /* Update histogram */
        const data_t val = in[win_pX + (win_pY * GHOST_AREA_WIDTH)];
        int *val_pos = &hist[val / HIST_SIZE];
        atomicAdd(val_pos, 1);
    }

    /* Search median (! only the master thread of each group !) */
    if (th_id == 0)
    {
        for (int i=0; i<HIST_SIZE; i++) {
            count += hist[i];
            if (count > MEDIAN_POSITION)
            {
                sh_base[th_group] = (data_t)i * HIST_SIZE;
                count -= hist[i];
                break;
            }
        }
    }

    __syncwarp(); // synchronize threads in warp
    const data_t base = sh_base[th_group];

    /* Reinit histogram */
    init_hist(hist, th_id);

    /* Compute histogram for lesser 8 bits */
    for (int i=th_id; i<WINDOW_SIZE; i+=TH_PER_GROUP) {
        // window pixel coords
        const int win_pX = (i % WINDOW_L) + pixelX;
        const int win_pY = (i / WINDOW_L) + pixelY;

        /* Update histogram IF 'val' has the same first 8 bits as 'base' */
        const data_t val = in[win_pX + (win_pY * GHOST_AREA_WIDTH)];
        data_t val_base = (val / HIST_SIZE) * HIST_SIZE;
        int *val_pos = &hist[val - val_base];
        if (val_base == base)
            atomicAdd(val_pos, 1);
    }
    /* Search median (! only the master thread of each group !) */
    if (th_id == 0)
    {
        for (int i=0; i<HIST_SIZE; i++) {
            count += hist[i];
            if (count > MEDIAN_POSITION)
            {
                /* Write median to buffer */
                out[pixelX + (pixelY * WIDTH)] = (data_t)i + base;
                break;
            }
        }
    }
}

int main( int argc, char *argv[] )
{
    if (argc != 6) {
        fprintf(stderr, "Usage: %s filein fileout width height radius\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int WIDTH = atoi(argv[3]);
    const int HEIGHT = atoi(argv[4]);
    const int RADIUS = atoi(argv[5]);

    if (WIDTH <= 0 || HEIGHT <= 0 || RADIUS <= 0) {
        fprintf(stderr, "FATAL: width, height and radius must be > 0\n");
        return EXIT_FAILURE;
    }

    FILE* filein = fopen(argv[1], "r");
    if (filein == NULL) {
        fprintf(stderr, "FATAL: can not read \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }

    const size_t IMG_SIZE = WIDTH * HEIGHT * sizeof(data_t);
    data_t *img = (data_t*)malloc(IMG_SIZE); assert(img != NULL);
    const size_t nread = fread(img, sizeof(data_t), WIDTH*HEIGHT, filein);
    assert(nread == WIDTH*HEIGHT);
    fclose(filein);

    /*** CUDA initialize ***/
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    omp_set_num_threads(num_devices);
    
    /* Use threads to launch the computation on all devices */
    #pragma omp parallel
    {
        const int id = omp_get_thread_num();
        cudaSetDevice(id);

        data_t *d_in;           // input image
        data_t *d_out;          // output image
        /* The input image is bigger, in order to initialize the ghost area */
        const int GHOST_AREA_WIDTH = (2 * RADIUS) + WIDTH;
        const int GHOST_AREA_HEIGHT = (2 * RADIUS) + HEIGHT;
        cudaSafeCall( cudaMalloc((void**)&d_in, GHOST_AREA_WIDTH * GHOST_AREA_HEIGHT * sizeof(data_t)) );
        cudaSafeCall( cudaMalloc((void**)&d_out, WIDTH * HEIGHT * sizeof(data_t)) );

        /* Initialize the ghost area */
        cudaSafeCall( cudaMemcpy(d_out, img, WIDTH * HEIGHT * sizeof(data_t), cudaMemcpyHostToDevice) );
        const int INIT_GRIDDIM = ((GHOST_AREA_WIDTH * GHOST_AREA_HEIGHT) + BLKDIM - 1) / BLKDIM;
        init_ghost_area<<< INIT_GRIDDIM, BLKDIM >>>(d_out, d_in, WIDTH, HEIGHT, RADIUS);
        cudaCheckError();

        /*** Start computation ***/        
        const int NUM_BLOCKS = ((WIDTH * HEIGHT) + THREAD_GROUPS - 1) / THREAD_GROUPS;      // total number of blocks to launch

        const int pstart = (NUM_BLOCKS * id) / num_devices;
        const int pend = (NUM_BLOCKS * (id+1)) / num_devices;

        const int GRID_SIZE = pend - pstart;                                                // number of blocks for this device

        const double tstart = hpc_gettime();
        median_filter_kernel<<< GRID_SIZE, BLKDIM >>>(d_in, d_out, WIDTH, HEIGHT, RADIUS, pstart * THREAD_GROUPS);
        cudaCheckError();
        const double elapsed = hpc_gettime() - tstart;
        fprintf(stderr, "[GPU%d] Execution time: %f\n", id, elapsed);

        /* Copy the portion computed by the device */
        int RESULT_SIZE = GRID_SIZE * THREAD_GROUPS * sizeof(data_t);                       // size of the computed portion
        if (id == num_devices-1)
        {
            RESULT_SIZE -= NUM_BLOCKS * THREAD_GROUPS * sizeof(data_t) - IMG_SIZE;          // avoids forbidden memory access
        }
        cudaSafeCall( cudaMemcpy(&img[pstart * THREAD_GROUPS], &d_out[pstart * THREAD_GROUPS], RESULT_SIZE, cudaMemcpyDeviceToHost) );
        
        /* Free memory */
        cudaFree(d_in);
        cudaFree(d_out);
    }
    

    FILE* fileout = fopen(argv[2], "w");
    if (fileout == NULL) {
        fprintf(stderr, "FATAL: can not create \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }

    const size_t nwritten = fwrite(img, sizeof(data_t), WIDTH*HEIGHT, fileout);
    assert(nwritten == WIDTH*HEIGHT);
    fclose(fileout);

    free(img);

    return EXIT_SUCCESS;
}