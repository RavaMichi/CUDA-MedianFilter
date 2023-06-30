/****************************************************************************
 *
 * cuda-median-qsel.cu - Image denoising using median filter
 *
 ****************************************************************************/

/***
% HPC - Image denoising using median filter
% Michele Ravaioli <michele.ravaioli3@unibo.it>
% Last modified: 2023-06-30

The file [cuda-median-qsel.cu](cuda-median-qsel.cu) contains a CUDA implementation
of an _image denoising_ algorithm which uses the _quickselection_.
The algorithm replaces the color of each pixel with the _median_ of a neighborhood
of radius `RADIUS` (including itself).

To compile:

        nvcc cuda-median-qsel.cu -o cuda-median-qsel -D WIDTH=width -D HEIGHT=height -D RADIUS=radius

To execute:

        ./cuda-median-qsel filein fileout

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

- [cuda-median-qsel.cu](cuda-median-qsel.cu)
- [hpc.h](hpc.h)

 ***/

#include "hpc.h" /* MUST be the first file included */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#ifndef RADIUS
#define RADIUS 20
#endif
#ifndef WIDTH
#define WIDTH 1500
#endif
#ifndef HEIGHT
#define HEIGHT 1500
#endif
#define BLKDIM 1024
#define GRIDDIM (((WIDTH * HEIGHT) + BLKDIM - 1) / BLKDIM)
#define HALO_WIDTH (2 * RADIUS + WIDTH)
#define HALO_HEIGHT (2 * RADIUS + HEIGHT)
#define STENCIL_SIZE (2 * RADIUS + 1)
#define REPLICATE

typedef uint16_t data_t;

#define ELEM_SWAP(a,b) { register data_t t=(a);(a)=(b);(b)=t; }

/*
 * Quickselect algorithm
 */
__device__ data_t quick_select(data_t arr[], int n) 
{
    int low, high ;
    int median;
    int middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[median] ;
        }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;

    /* Re-set active partition */
    if (hh <= median)
        low = ll;
    if (hh >= median)
        high = hh - 1;
    }
}

/*
 * Ghost Area Initialization Kernel
 */
__global__ void init_ghost_area( data_t* in, data_t* out )
{
    /* coords of selected pixel */
    const int IDX = (threadIdx.x + blockIdx.x * BLKDIM) % HALO_WIDTH;
    const int IDY = (threadIdx.x + blockIdx.x * BLKDIM) / HALO_WIDTH;
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

    /* fill buffer */
    out[IDX + IDY*HALO_WIDTH] = in[x + y*WIDTH];
}

/*
 * Median Filtering using Quickselection
 */
__global__ void median_filter_kernel( data_t* in, data_t* out )
{
    /* coordinate del pixel in elaborazione */
    const int IDX = (threadIdx.x + blockIdx.x * BLKDIM) % WIDTH;
    const int IDY = (threadIdx.x + blockIdx.x * BLKDIM) / WIDTH;

    if (IDY >= HEIGHT) return;

    data_t tmp[STENCIL_SIZE * STENCIL_SIZE];

    for (int y=0; y<STENCIL_SIZE; y++) {
        /* copia lo stencil in tmp */
        for (int x=0; x<STENCIL_SIZE; x++) {
            tmp[x + y*STENCIL_SIZE] = in[(IDX + x) + (IDY + y)*HALO_WIDTH];
        }
    }
    /* calcola la mediana */
    out[IDX + IDY*WIDTH] = quick_select(tmp, STENCIL_SIZE*STENCIL_SIZE);
}

int main( int argc, char *argv[] )
{
    const size_t IMG_SIZE = WIDTH * HEIGHT * sizeof(data_t);

    if (argc != 3) {
        fprintf(stderr, "Usage: %s filein fileout\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE* filein = fopen(argv[1], "r");
    if (filein == NULL) {
        fprintf(stderr, "FATAL: can not read \"%s\"\n", argv[1]);
        return EXIT_FAILURE;
    }

    data_t *img = (data_t*)malloc(IMG_SIZE); assert(img != NULL);
    const size_t nread = fread(img, sizeof(data_t), WIDTH*HEIGHT, filein);
    assert(nread == WIDTH*HEIGHT);
    fclose(filein);

    /* CUDA initialize */
    data_t *in, *out;
    cudaSafeCall( cudaMalloc((void**)&in, HALO_WIDTH * HALO_HEIGHT * sizeof(data_t)) );
    cudaSafeCall( cudaMalloc((void**)&out, WIDTH * HEIGHT * sizeof(data_t)) );

    cudaMemcpy(out, img, WIDTH * HEIGHT * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaCheckError();

    const int INIT_GRIDDIM = ((HALO_WIDTH * HALO_HEIGHT) + BLKDIM - 1) / BLKDIM;
    init_ghost_area<<< INIT_GRIDDIM, BLKDIM >>>(out, in);
    cudaCheckError();

    /* Init computation */
    const double tstart = hpc_gettime();
    median_filter_kernel<<< GRIDDIM, BLKDIM >>>(in, out);
    cudaCheckError();
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Execution time: %f\n", elapsed);

    FILE* fileout = fopen(argv[2], "w");
    if (fileout == NULL) {
        fprintf(stderr, "FATAL: can not create \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }

    cudaSafeCall( cudaMemcpy(img, out, WIDTH * HEIGHT * sizeof(data_t), cudaMemcpyDeviceToHost) );
    const size_t nwritten = fwrite(img, sizeof(data_t), WIDTH*HEIGHT, fileout);
    assert(nwritten == WIDTH*HEIGHT);
    fclose(fileout);

    free(img);

    return EXIT_SUCCESS;
}
