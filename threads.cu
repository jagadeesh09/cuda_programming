/*

        Example of using threads in a CUDA program


                                                    */

#include <stdio.h>
#include <time.h>


#define N 512



__global__ void add(int *a, int *b, int *c){

    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];

}

void random_ints(int *a, int n)
{

    int i;
    for (i=0; i < n; ++i){

        a[i] = rand();

    }

}
int main(void) {
    
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    random_ints(b, N);
    random_ints(a, N);

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    
    clock_t t;
    t = clock();

    add<<<1,N>>>(d_a, d_b, d_c);

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("Time taken by function is %f seconds\n",time_taken);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_c);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);
    return 0;

}
