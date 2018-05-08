/*

        Example of using threads and blocks in a CUDA program


                                                    */

#include <stdio.h>
#include <time.h>


#define N 8732
#define classes 2

void random_float(float *a, int n)
{

    int i;
    for (i=0; i < n; ++i){

        a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    }

}
__global__ void Softmax(float *x, int channels, float *y){

    __shared__ float sum_val;
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    float val = x[index];
    float exp_val = __expf(val);

    atomicAdd(&sum_val, exp_val);
    __syncthreads();

    y[index] = __fdiv_rd(exp_val,sum_val);


}
int main(void) {
    
    float *a; float *c;
    float *d_a; float *d_c;
    int size = N * sizeof(float);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_c, size);

    a = (float *)malloc(size);
    c = (float *)malloc(size);

    //random_ints(b, N);
    random_float(a, N);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
    
    clock_t t;
    t = clock();

    Softmax<<<N/classes, classes>>>(d_a, classes, d_c);

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("Time taken by function is %f seconds\n",time_taken);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    int k;
    for(k = 0; k < N  ;k++){
        printf("%f ",c[k]);

    }
    cudaFree(d_a);
    cudaFree(d_c);

    free(a);
    free(c);
    return 0;

}
