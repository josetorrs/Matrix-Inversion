#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <assert.h>
#include <sys/time.h>
#define Tix 32
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

__global__ void GaussJordan_gpu(float *Aaug, float *subpivot, int N, int iter) {
    int c = iter + blockIdx.x * Tix*Tix;
    int r = blockIdx.y;
    float scale;
    int ti = threadIdx.x;
    scale =  Aaug[r*2*N+iter];
    __shared__ float col[Tix*Tix];
    __shared__ float colj[Tix*Tix];


    if (r != iter){
        if (c + ti < 2*N){
           col[ti] = Aaug[iter*2*N+c+ti];  
           colj[ti] = Aaug[r*2*N+c+ti]; 
           colj[ti] -= scale*col[ti];
           Aaug[r*2*N+c+ti] = colj[ti]; 
        }
        if (blockIdx.x == 0){
            if (ti == 0){
                subpivot[r] = Aaug[r*2*N+iter+1];
            }
        }
    }
}

__global__ void update_row_gpu(float *Aaug, float *subpivot, int N, int iter, int use) {
    int c = iter + blockIdx.x * Tix*Tix;
    int ti = threadIdx.x;
    if (c + ti < 2*N){
        Aaug[iter*2*N+c+ti] += Aaug[use*2*N+c+ti];
    }
    if (blockIdx.x == 0){
        if (ti == 0){
            subpivot[iter] = Aaug[iter+iter*2*N];
        }
    }
}

__global__ void scale_row_gpu(float *Aaug, float *subpivot, int N, int iter) {
    int c = iter + blockIdx.x * Tix*Tix;
    int ti = threadIdx.x;
    if (c + ti < 2*N){
        Aaug[iter*2*N+c+ti] = Aaug[iter*2*N+c+ti]/subpivot[iter];
    }
}

int main(int argc, char *argv[]){
    float *Aaug, *Aaug_cu, *subpivot, *subpivot_cu;
    int iter, m, i, j,  N, use;
    FILE * f;

    // Checks to see valid number of inputs given 
    if (argc != 3)
    {
        printf("need input and N\n");
        return -1;
    }
    N = strtol(argv[2],NULL,10);


    // Checks to see if a valid .mtx file was given
    int memSize = 2*N*N*sizeof(float);
    Aaug = (float *)malloc(memSize);	
    subpivot = (float *)malloc(N*sizeof(float));	

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
        subpivot[i] = Aaug[i*2*N];
    }
    fclose(f);

    cudaMalloc((void**)&Aaug_cu, memSize);
    cudaMalloc((void**)&subpivot_cu, N*sizeof(float));
    
    cudaMemcpy(Aaug_cu, Aaug, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(subpivot_cu, subpivot, N*sizeof(float), cudaMemcpyHostToDevice);


    // Runs GPU Code
    int bn, rn;
    dim3 nblocks, nthreads, nblocks_1, nthreads_1;
    nthreads.x = Tix*Tix;
    nthreads.y =  1;
    nthreads_1.x = Tix*Tix;
    nthreads_1.y = 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (iter=0;iter<N; iter++){
        bn = MAX((2*N-iter)/(Tix*Tix),1);             // Defines number of subdivisions in the row
        rn = N;                                       // Defines how many rows to update
        
        nblocks.x = bn;
        nblocks.y = rn;

        nblocks_1.x = bn;
        nblocks_1.y = 1;

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
        if(iter<N){        // Won't perform reduction if iter = N (at the bottom) 
            GaussJordan_gpu<<<nblocks, nthreads>>>(Aaug_cu, subpivot_cu, N, iter);
            cudaDeviceSynchronize();
            cudaMemcpy(subpivot, subpivot_cu, N*sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf ("gpu time: %f ms\n", milliseconds);
    cudaMemcpy(Aaug, Aaug_cu, memSize, cudaMemcpyDeviceToHost);
    return 0;
}
