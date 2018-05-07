/*

        Example of using threads and blocks in a CUDA program


                                                    */

#include <stdio.h>
#include <time.h>


#define N 16
#define THREADS_PER_BLOCK 8
#define BLOCK_SIZE 5
#define RADIUS 2

__global__ void add(int *a, int *b, int *c){

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];

}

void random_ints(int *a, int n)
{

    int i;
    for (i=0; i < n; ++i){

        a[i] = rand();

    }

}

__global__ void stencil_1d(int *in, long int *out){

    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int centre_index = threadIdx.x + blockIdx.x * blockDim.x;
    int last_index = threadIdx.x + RADIUS;

    temp[last_index] = in[centre_index];

    if(threadIdx.x < RADIUS) {

        temp[last_index - RADIUS] = in[centre_index - RADIUS];
        temp[last_index + BLOCK_SIZE] = in[centre_index + BLOCK_SIZE];

    }

    //__syncthreads();
    printf("blockId : %d block dim: %d thread: %d last: %d centre %d       \n",blockIdx.x,blockDim.x,threadIdx.x,last_index,centre_index);
    long int result = 0;
    for ( int offset = -RADIUS ; offset <= RADIUS ; offset++){
            printf("%d ", temp[last_index + offset]);
            result += temp[last_index + offset];
            //printf("-------------------------------");

    }

    printf("-------------------------------\n");
    out[centre_index] = result;
    printf("%ld\n",result);


}
int main(void) {
    
    int *a; long int *b, *c;
    int *d_a;long int *d_b, *d_c;
    int size = N * sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, N*sizeof(long int));
    cudaMalloc((void **)&d_c, N*sizeof(long int ));

    a = (int *)malloc(size);
    b = (long *)malloc(N*sizeof(long int));
    c = (long *)malloc(N*sizeof(long int));

    //random_ints(b, N);
    random_ints(a, N);
    int i;
    for (i=0;i<N; i++){
        printf("%d\n",a[i]);
    }
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, b, size, cudaMemcpyHostToDevice);
    
    clock_t t;
    t = clock();

    stencil_1d<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_c);

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
