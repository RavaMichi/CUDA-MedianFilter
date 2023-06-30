# Sviluppo di codice CUDA per l'ottimizzazione del filtro mediano su immagini 2D
Tesi triennale del Corso di Laurea in Ingegneria e Scienze informatiche
Anno Accademico 2022/2023
_Presentata da_: Ravaioli Michele
_Relatore_: Marzolla Moreno

# Compilazione ed esecuzione

Per la compilazione dei codici presi in considerazione nell'ambiente Linux, è indispensabile avere installato il compilatore GCC per la compilazione del codice omp-median.c e il compilatore NVCC per la compilazione dei codici CUDA/C. Inoltre, per l'esecuzione dei codici CUDA, è necessario che il sistema sia dotato di almeno una scheda grafica NVIDIA adeguata.

- **omp-median.c**
  Per compilare:
  `gcc -std=c99 -fopenmp -Wall -Wpedantic -O2 omp-median.c -o omp-median`
  Per eseguire:
  `./omp-median -a quicksort -w width -h height -r radius -o fileout filein`

- **cuda-median-qsel.cu**
  Per compilare (dimensione e raggio vanno specificati alla compilazione):
  `nvcc cuda-median-qsel.cu -o cuda-median-qsel -D WIDTH=width -D HEIGHT=height -D RADIUS=radius`
  Per eseguire:
  `./cuda-median-qsel filein fileout`

- **cuda-median-hist.cu**
  Per compilare:
  `nvcc cuda-median-hist.cu -o cuda-median-hist`
  Per eseguire:
  `./cuda-median-hist filein fileout width height radius`

- **cuda-median-multi.cu**
  Per compilare:
  `nvcc cuda-median-multi.cu -o cuda-median-multi -Xcompiler -fopenmp`
  Per eseguire:
  `./cuda-median-hist filein fileout width height radius`
