/****************************************************************************
 *
 * omp-median.c - Image denoising using median filter
 *
 * Copyright 2018--2023 Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% HPC - Image denoising using median filter
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last modified: 2023-05-23

The file [omp-median.c](omp-median.c) contains a serial implementation
of an _image denoising_ algorithm that (to some extent) can be used to
"cleanup" color images. The algorithm replaces the color of each pixel
with the _median_ of a neighborhood of radius `RADIUS` (including
itself).

To compile:

        gcc -std=c99 -fopenmp -Wall -Wpedantic -O2 omp-median.c -o omp-median

To execute:

        ../omp-median -a hist_byrow -w 2729 -h 3580 -r 41 -o out.raw datasets/MG-RAW_15165_R_MLO_WT_2729-HT_3580.raw

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

- [omp-median.c](omp-median.c)
- [hpc.h](hpc.h)

 ***/
#include "hpc.h" /* MUST be the first file included */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include <omp.h>

typedef uint16_t data_t;
const size_t DATA_SIZE = sizeof(data_t);
const size_t HIST_LEN = 1<<(8*DATA_SIZE - 1);

#define REPLICATE

int IDX(int i, int j, int cols, int rows)
{
#ifdef REPLICATE
    i = (i<0 ? 0 : (i>=rows ? rows-1 : i));
    j = (j<0 ? 0 : (j>=cols ? cols-1 : j));
#else
    i = (i + rows) % rows;
    j = (j + cols) % cols;
#endif
    return (i*cols + j);
}

void swap(data_t *v, int i, int j)
{
    const data_t tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
}

/**
 * Return a random integer in [a, b]
 */
int randab(int a, int b)
{
    return a + (rand() % (b-a+1));
}

/**
 * The call:
 *
 * k = partition(v, start, end);
 *
 * rearranges the content of v[start..end] such that:
 *
 * - v[i] <= v[k] for all i = start..k-1
 * - v[j] > v[k] for all  j = k+1..end
 */
int partition(data_t *v, int start, int end)
{
    /* L'invariante della procedura partition() descritta nel libro
       è la seguente:

       - v[k] <= pivot per ogni start <= k <= i
       - v[k] > pivot per ogni  i+1 <= k < j

    */
    // swap(v, randab(start, end), end);
    const data_t pivot = v[end];
    int i = (start - 1), j;

    for (j = start; j < end; j++) {
        if (v[j] <= pivot) {
            i++;
            swap(v, i, j);
        }
    }

    swap(v, i+1, end);
    return i + 1;
}

int quickselect_rec(data_t *v, int start, int end, int k)
{
    assert(start <= end);
    assert(start <= k && k <= end);
    const int split = partition(v, start, end);
    if (k == split)
        return v[k];
    else if (k < split)
        return quickselect_rec(v, start, split - 1, k);
    else
        return quickselect_rec(v, split + 1, end, k);
}

int compare(const void* a, const void *b)
{
    const data_t val_a = *(const data_t*)a;
    const data_t val_b = *(const data_t*)b;
    return val_a - val_b;
}

/**
 * Compute the median of `v[]` if length `n`. The median is the value
 * that would occupy position `n/2` if `v[]` were sorted.
 */
data_t median(data_t *v, int n)
{
#if 1
    return quickselect_rec(v, 0, n-1, n/2);
#else
    /* Dato che ottenevo valori diversi da quelli attesi, ho provato a
       fare come fa il codice fornito, cioè ordinare l'array v[] e poi
       prendere l'elemento in posizione centrale. Ottengo lo stesso
       risultato (ma in modo meno efficiente) rispetto alla funzione
       quickselect_ret(). */
    qsort(v, n, DATA_SIZE, compare);
    return v[n/2];
#endif
}
void median_filter_quicksort( int radius, data_t *bmap, int width, int height )
{
    const size_t TMP_LEN = (2*radius+1) * (2*radius+1);
    const size_t BMAP_SIZE = width*height*DATA_SIZE;
    data_t *out = (data_t*)malloc(BMAP_SIZE);
    assert(out != NULL);

#if __GNUC__ < 9
#pragma omp parallel default(none) shared(width, height, bmap, out, radius)
#else
#pragma omp parallel default(none) shared(width, height, bmap, out, radius, TMP_LEN)
#endif
    {
        data_t *tmp = (data_t*)malloc(TMP_LEN * DATA_SIZE);
        assert(tmp != NULL);
#pragma omp for collapse(2) schedule(dynamic)
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                int k = 0;
                for (int di=-radius; di<=radius; di++) {
                    for (int dj=-radius; dj<=radius; dj++) {
                        tmp[k++] = bmap[IDX(i+di, j+dj, width, height)];
                    }
                }
                assert(k == TMP_LEN);
                qsort(tmp, TMP_LEN, DATA_SIZE, compare);
                out[IDX(i, j, width, height)] = tmp[TMP_LEN/2];
            }
        }
        free(tmp);
    }
    memcpy(bmap, out, BMAP_SIZE);
    free(out);
}

/**
 * Apply the median filter algorithm to each pixel of image `bmap` of
 * size `width` x `height` and window of radius `radius`. Use the
 * quickselect algorithm for computing the median.
 */
void median_filter_quickselect( int radius, data_t *bmap, int width, int height )
{
    const size_t TMP_LEN = (2*radius+1) * (2*radius+1);
    const size_t BMAP_SIZE = width*height*DATA_SIZE;
    data_t *out = (data_t*)malloc(BMAP_SIZE);
    assert(out != NULL);

#if __GNUC__ < 9
#pragma omp parallel default(none) shared(width, height, bmap, out, radius)
#else
#pragma omp parallel default(none) shared(width, height, bmap, out, radius, TMP_LEN)
#endif
    {
        data_t *tmp = (data_t*)malloc(TMP_LEN * DATA_SIZE);
        assert(tmp != NULL);
#pragma omp for collapse(2) schedule(dynamic)
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                int k = 0;
                for (int di=-radius; di<=radius; di++) {
                    for (int dj=-radius; dj<=radius; dj++) {
                        tmp[k++] = bmap[IDX(i+di, j+dj, width, height)];
                    }
                }
                assert(k == TMP_LEN);
                out[IDX(i, j, width, height)] = median(tmp, TMP_LEN);
            }
        }
        free(tmp);
    }
    memcpy(bmap, out, BMAP_SIZE);
    free(out);
}

/**
 * Set to zero the content of histogrma `hist`
 */
void clear_histogram(int *hist)
{
    for (data_t k=0; k<HIST_LEN; k++)
        hist[k] = 0;
}

/**
 * Compute the histogram of the values located within a window of
 * radius `radius` centered at (i,j). The array `hist` must have
 * length exactly (1<< (8*DATA_SIZE - 1)) and must have been zeroed by
 * the caller.
 */
void fill_histogram(const data_t *bmap, int i, int j, int radius, int width, int height, int *hist)
{
    for (int di=-radius; di<=radius; di++) {
        for (int dj=-radius; dj<=radius; dj++) {
            const data_t val = bmap[IDX(i+di, j+dj, width, height)];
            hist[val]++;
        }
    }
}

/**
 * Given an already filled histogram of a window of rasiud `radius`
 * centered at (i, j), update the histogram by shifting the window one
 * position to the right
 */
void shift_histogram(const data_t *bmap, int i, int j, int radius, int width, int height, int *hist)
{
    for (int di=-radius; di<=radius; di++) {
        const data_t val_left = bmap[IDX(i+di, j-radius, width, height)];
        hist[val_left]--;
        const data_t val_right = bmap[IDX(i+di, j+radius+1, width, height)];
        hist[val_right]++;
    }
}

/**
 * Given an already filled histogram `hist` of length (1<<(8*DATA_SIZE
 * - 1) return the index corresponding to the median
 * value. Specifically, let `n` be the total sum of the
 * histogram. Then, return the maximum index `k` such that (sum
 * hist[0..k]) <= n/2
 */
data_t hist_median(const int *hist, int radius)
{
    const int n = (2*radius+1) * (2*radius+1); // total n. of elements in the histogram
    data_t k = 0;
    int count = 0;

    do {
        count += hist[k];
        k++;
    } while (count <= n/2 && k < HIST_LEN);
    assert(k>=1 && k<=HIST_LEN);
    return k-1;
}

/**
 * Apply the median filter algorithm to each pixel of image `bmap` of
 * size `width` x `height` and window of radius `radius`. Use the
 * histogram-based algorithm for computing the median. The histogram
 * is computed from scratch for each pixel.
 */
void median_filter_hist( int radius, data_t *bmap, int width, int height )
{
    const size_t BMAP_SIZE = width * height * DATA_SIZE;
    data_t *out = (data_t*)malloc(BMAP_SIZE);
    assert(out != NULL);

#if __GNUC__ < 9
#pragma omp parallel default(none) shared(width, height, bmap, out, radius)
#else
#pragma omp parallel default(none) shared(width, height, bmap, out, radius, HIST_LEN)
#endif
    {
        int *hist = (int*)malloc(HIST_LEN*sizeof(int));
        assert(hist != NULL);
#pragma omp for collapse(2) schedule(static,32)
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                clear_histogram(hist);
                fill_histogram(bmap, i, j, radius, width, height, hist);
                out[IDX(i, j, width, height)] = hist_median(hist, radius);
            }
        }
        free(hist);
    }
    memcpy(bmap, out, BMAP_SIZE);
    free(out);
}

/**
 * Apply the median filter algorithm to each pixel of image `bmap` of
 * size `width` x `height` and window of radius `radius`. Use the
 * histogram-based algorithm for computing the median. The histogram
 * is *not* computed from scratch for each pixel; instead, when the
 * window is shifted, the old histogram is updated.
 */
void median_filter_hist_byrow( int radius, data_t *bmap, int width, int height )
{
    const size_t BMAP_SIZE = width * height * DATA_SIZE;
    data_t *out = (data_t*)malloc(BMAP_SIZE);
    assert(out != NULL);

#if __GNUC__ < 9
#pragma omp parallel default(none) shared(width, height, bmap, out, radius, stderr)
#else
#pragma omp parallel default(none) shared(width, height, bmap, out, radius, stderr, HIST_LEN)
#endif
    {
        int *hist = (int*)malloc(HIST_LEN*sizeof(int));
        assert(hist != NULL);
#pragma omp for schedule(static,32)
        for (int i=0; i<height; i++) {
            clear_histogram(hist);
            fill_histogram(bmap, i, 0, radius, width, height, hist);
            for (int j=0; j<width; j++) {
                out[IDX(i, j, width, height)] = hist_median(hist, radius);
                shift_histogram(bmap, i, j, radius, width, height, hist);
            }
        }
        free(hist);
    }
    memcpy(bmap, out, BMAP_SIZE);
    free(out);
}

typedef void (*median_filter_algo_t)( int radius,
                                      data_t *bmap,
                                      int width,
                                      int height );

int main( int argc, char *argv[] )
{
    int width = -1, height = -1, radius = 41;
    const char *infile = NULL, *outfile = "out.raw";
    int i, opt;

    struct {
        const char *name;
        const char *description;
        median_filter_algo_t fun;
    } median_filter_algos[] = { {"quickselect", "Quickselect-based median", median_filter_quickselect},
                                {"hist", "Histogram-based median", median_filter_hist},
                                {"hist_byrow", "Optimized histogram-based median", median_filter_hist_byrow},
                                {"sort", "Quicksort-based median", median_filter_quicksort},
                                {NULL, NULL, NULL}
    };

    const char *algo_name = median_filter_algos[0].name;
    median_filter_algo_t algo_fun = median_filter_algos[0].fun;

    assert(DATA_SIZE == 2);

    while ((opt = getopt(argc, argv, "Ha:w:h:r:o:")) != -1) {
        switch(opt) {
        case 'a':
            i = 0;
            while (median_filter_algos[i].name && strcmp(optarg, median_filter_algos[i].name)) {
                i++;
            }
            if (median_filter_algos[i].name) {
                algo_name = median_filter_algos[i].name;
                algo_fun = median_filter_algos[i].fun;
            } else {
                fprintf(stderr, "\nFATAL: invalid algorithm %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'w':
            width = atoi(optarg);
            break;
        case 'h':
            height = atoi(optarg);
            break;
        case 'r':
            radius = atoi(optarg);
            break;
        case 'o':
            outfile = optarg;
            break;
        case 'H':
            fprintf(stderr,
                    "Usage: %s [-H] [-a algo] [-w width] [-h height] [-r radius] [-o outfile] infile\n\n"
                    "-H\t\tprint help\n"
                    "-a algo\t\tset algorithm (see below)\n"
                    "-w width\timage width\n"
                    "-h height\timage height\n"
                    "-r radius\tfilter radius\n"
                    "-o outfile\toutput file name\n"
                    "infile\t\tinput file name\n\n"
                    "Valid algorithm names:\n\n", argv[0]);
            for (i=0; median_filter_algos[i].name; i++) {
                fprintf(stderr, "%-13s\t%s%s\n",
                        median_filter_algos[i].name,
                        median_filter_algos[i].description,
                        i == 0 ? " (default)" : "");
            }
            fprintf(stderr, "\n");
            return EXIT_SUCCESS;
        default:
            fprintf(stderr, "Unrecognized option %s\n", argv[optind]);
            return EXIT_FAILURE;
        }
    }

    if (width < 0 || height < 0) {
        fprintf(stderr, "You must specify width and height\n");
        return EXIT_FAILURE;
    }

    if (optind >= argc) {
        fprintf(stderr, "No input file given\n");
        return EXIT_FAILURE;
    }

    infile = argv[optind];

    FILE* filein = fopen(infile, "r");
    if (filein == NULL) {
        fprintf(stderr, "FATAL: can not open input file \"%s\"\n", infile);
        return EXIT_FAILURE;
    }

    const size_t IMG_SIZE = width * height * DATA_SIZE;

    data_t *img = (data_t*)malloc(IMG_SIZE); assert(img != NULL);
    const size_t nread = fread(img, DATA_SIZE, width*height, filein);
    assert(nread == width*height);
    fclose(filein);

    fprintf(stderr,
            "Input........... %s\n"
            "Width........... %d\n"
            "Height.......... %d\n"
            "Radius.......... %d\n"
            "Output.......... %s\n"
            "Algorithm....... %s\n",
            infile,
            width,
            height,
            radius,
            outfile,
            algo_name);
    const double tstart = hpc_gettime();
    algo_fun(radius, img, width, height);
    const double elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "Execution time.. %f\n", elapsed);

    FILE* fileout = fopen(outfile, "w");
    if (fileout == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", outfile);
        return EXIT_FAILURE;
    }

    const size_t nwritten = fwrite(img, DATA_SIZE, width*height, fileout);
    assert(nwritten == width*height);
    fclose(fileout);

    free(img);

    return EXIT_SUCCESS;
}
