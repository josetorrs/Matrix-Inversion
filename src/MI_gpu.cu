#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <assert.h>
#include <sys/time.h>

#define Tix 32

__global__ void GaussJordan_gpu(float *Aaug, float *subpivot, int N, int iter) {
    __shared__ float smem_a[Tix*Tix];
    int c = iter + blockIdx.x * Tix*Tix;
    int r = iter + blockIdx.y;
    float crat;
    int ti = threadIdx.x;
    smem_a[ti] = Aaug[iter*2*N + c + ti];
    crat =  Aaug[r*2*N+iter];

    if (c + ti < 2*N){
        Aaug[r*2*N+c+iter+threadIdx.x] -= crat*smem_a[threadIdx.x];
    }
    if (blockIdx.x == 0){
        if (threadIdx.x == 0){
            subpivot[r] = Aaug[r*2*N+c];
        }
        if (blockIdx.y == 0){
            if (threadIdx.x == 0){
                subpivot[iter] = Aaug[iter+iter*2*N];
            }
        }
    }
}

__global__ void update_row_gpu(float *Aaug, float *subpivot, int N, int iter, int use) {
    int c = iter + blockIdx.x * Tix*Tix;
    int ti = threadIdx.x;
    if (c + ti < 2*N){
        Aaug[iter*2*N+c+threadIdx.x] += Aaug[use*2*N+c+threadIdx.x];
    }
    if (blockIdx.x == 0){
        if (threadIdx.x == 0){
            subpivot[iter] = Aaug[iter+iter*2*N];
        }
    }
}

__global__ void scale_row_gpu(float *Aaug, float *subpivot, int N, int iter) {
    int c = iter + blockIdx.x * Tix*Tix;
    int ti = threadIdx.x;
    if (c + ti < 2*N){
        Aaug[iter*2*N+threadIdx.x] += Aaug[iter*2*N+threadIdx.x]/subpivot[iter];
    }
}

__global__ void Backsolve_gpu(float *Aaug, int N, int iter) {
    __shared__ float smem_a[Tix*Tix];
    int c = N + blockIdx.x * Tix*Tix;
    int r = blockIdx.y;
    float crat;
    int ti = threadIdx.x;
    smem_a[ti] = Aaug[iter*2*N + c + ti];
    crat =  Aaug[r*2*N+iter];
    if (c + ti < 2*N){
        Aaug[r*2*N+c+iter+threadIdx.x] -= crat*smem_a[threadIdx.x];
    }
}



int main(int argc, char *argv[]){
    float *Aaug, *Aaug_cu, *subpivot, *subpivot_cu;
    int iter, m, i, j,  N, use;
    FILE * f;

    // Checks to see valid number of inputs given 
    if (argc > 4 || argc < 4){
        printf("Error: expected 3 input (A .mtx file, Ainv .mtx file, N), given %d\n", argc-1);
        exit(-1);
    }
    N = strtol(argv[3],NULL,10);


    // Checks to see if a valid .mtx file was given
    int memSize = 2*N*N*sizeof(float);
    Aaug = (float *)malloc(memSize);	
    subpivot = (float *)malloc(N);	

    f = fopen(argv[1], "rb");
    for (i=0; i<N; i++){
        for (j=0; j<N; j++){
            fscanf(f, "%f", &Aaug[2*N*i+j]);
            if (i==j){
                Aaug[2*N*i+N+j] = 1;
            }
            else{
                Aaug[2*N*i+N+j] = 0;
            }
        }

    }
    fclose(f);

    cudaMalloc((void**)&Aaug_cu, memSize);
    cudaMalloc((void**)&subpivot_cu, N);
    cudaMemcpy(Aaug, Aaug_cu, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(subpivot, subpivot_cu, memSize, cudaMemcpyHostToDevice);


    // Runs GPU Code
    dim3 nblocks(2*N/(32*32), N);
    dim3 nthreads(32*32, 1);

    dim3 nblocks_1(2*N/(32*32), 1);
    dim3 nthreads_1(32*32, 1);

    for (iter=0;iter<N; iter++){
        cudaMemcpy(subpivot_cu, subpivot, memSize, cudaMemcpyDeviceToHost);
                                       
        if (sqrt(subpivot[iter]*subpivot[iter])<.00000000000001){      // checks for invertability
            for (m=1; m+iter<N; m++){                // loops through lower rows for nonzero in pivot
                if (sqrt(subpivot[iter+m]*subpivot[iter+m])>.000000000000001){   // checks if nonzero pivot
                    use = m+iter;
                    goto update;                // exits if pivot found
                }
                else if(m==N-1){
                    printf("Error matrix not invertible \n"); // if no pivot found, not inverable
                    exit(-1);
                }
            }
            printf("Error matrix not invertible \n");         // if at the last pivot and zero, not invertable
            exit(-1);
            update: update_row_gpu<<<nblocks_1, nthreads_1>>>(Aaug_cu, subpivot_cu, N, iter, use);
            cudaDeviceSynchronize();
        }
        scale_row_gpu<<<nblocks_1, nthreads_1>>>(Aaug_cu, subpivot_cu, N, iter);
        cudaDeviceSynchronize();
        
        GaussJordan_gpu<<<nblocks, nthreads>>>(Aaug_cu, subpivot_cu, N, iter);
        cudaDeviceSynchronize();
    }

    for (iter=N-1;iter>=0; iter--){
        Backsolve_gpu<<<nblocks, nthreads>>>(Aaug_cu, N, iter);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(Aaug_cu, Aaug, memSize, cudaMemcpyDeviceToHost);

    // Writes output to file (for testing) blocking
    FILE *of2 = fopen(argv[2], "wb");
    for (i=0; i<N; i++){
        for(j=N; j<2*N; j++){
            fprintf(of2, "%f ", Aaug[N+2*N*i+j]);
        }
        fprintf(of2, "\n");
    }
    fclose(of2);
}
