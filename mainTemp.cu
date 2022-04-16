#include <stdio.h>
#include <time.h>
#include "support.h"
#include "Kernel1.cu"
#include "Kernel2.cu"
#include "Kernel3.cu"
#include "generate_instance.h"
#ifndef max
    #define max(a,b) ((a) > (b) ? (a) : (b))
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct {
    unsigned int w_hat, p_hat, ub, lb, s;
} Node;

struct Instance Inst;
int numberOfItems, q_size;


int Capacity;

unsigned int bnb_lb;
int j, new_q_size;

void *fixed_cudaMalloc(size_t len)
{
    void *p;
    if (cudaMalloc(&p, len) == cudaSuccess) return p;
    return 0;
}

void copy_to_device(int* h_arr, int* d_arr, int len){

    cudaError_t cuda_ret = cudaMemcpy(d_arr, h_arr, sizeof(int)*len, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");
    gpuErrchk( cudaMalloc(&d_arr, len*sizeof(int)) );
}

int MAX = 120;
int* W_star = (int*) malloc( MAX*sizeof(int) );
int* P_star = (int*) malloc( MAX*sizeof(int) );
int *W_star_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX);
int *P_star_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX);

int main(int argc, char**argv) {

    clock_t start_t, end_t;
    double diff_t;
    
    cudaError_t cuda_ret;

    int maxVal = 1000;
    
    srand(1);
    numberOfItems = 10;
    
    
    double inst_time = 0;
    printf("Testing instances of size = %d \n", numberOfItems);
    
    Inst = getInstance(numberOfItems, maxVal); // Get Instance data
    for (int iter = 0; iter < numberOfItems; iter++)
    {
        W_star[iter] = Inst.weight[iter];
        P_star[iter] = Inst.price[iter];
    }
    Capacity = Inst.capacity;

    start_t = clock();
    
    printf("\nSetting up the problem..."); fflush(stdout);
    // startTime(&timer);

    unsigned int n;
    n = numberOfItems;

    // ALLOCATING HOST MEMORY ----------------------------------------------
    int* W_h = (int*) malloc( MAX*sizeof(int) );
    int* P_h = (int*) malloc( MAX*sizeof(int) );
    int* S_h = (int*) malloc( MAX*sizeof(int) );
    int* U_h = (int*) malloc( MAX*sizeof(int) );
    int* L_h = (int*) malloc( MAX*sizeof(int) );
    // int* C_bar_h = (int*) malloc( 1*sizeof(int) );

    int w = 0, p = 0, s = -1;
    int C_bar;
    
    while (w <= Capacity){
        s++; // initial slack
        w += W_star[s];
        p += P_star[s];
    }

    copy_to_device(W_star, W_star_d, MAX);
    copy_to_device(P_star, P_star_d, MAX);
    // ATTRIBUTES OF ROOT NODE: Initialize host variables ----------------------------------------------
    S_h[0] = s;
    P_h[0] = (p - P_star[s]);
    W_h[0] = (w - W_star[s]);
    C_bar = Capacity - (w - W_star[s]);
    U_h[0] = (p - P_star[s]) + (C_bar * P_star[s]/(double)W_star[s]);

    int wx = 0;
    bnb_lb = (p - P_star[s]);
    for (int i = s+1; i < numberOfItems; i++){
        if (wx + W_star[i] <= C_bar){
            wx += W_star[i];
            bnb_lb += P_star[i];
        }
    }
    L_h[0] = bnb_lb;
    
    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Number of Items = %u\n", numberOfItems);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    // startTime(&timer);

    q_size = 1; // initialized to 1 because there is only one node i.e. root node
    //new_q_size = q_size;
    int curr_pos = 0;

    int* W_d;
    cuda_ret = cudaMalloc((void**) &W_d, sizeof(int)*MAX);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    
    int* P_d;
    cuda_ret = cudaMalloc((void**) &P_d, sizeof(int)*MAX);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    int* S_d;
    cuda_ret = cudaMalloc((void**) &S_d, sizeof(int)*MAX);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    int* U_d;
    cuda_ret = cudaMalloc((void**) &U_d, sizeof(int)*MAX);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    int* L_d;
    cuda_ret = cudaMalloc((void**) &L_d, sizeof(int)*MAX);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    int* C_bar_d;
    cuda_ret = cudaMalloc((void**) &C_bar_d, sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    int *k_d = (int*) fixed_cudaMalloc(sizeof(int));
    
    int *q_d = (int*) fixed_cudaMalloc(sizeof(int));

    cudaDeviceSynchronize();
    printf("Device variables Allocated ..."); fflush(stdout);

    for (int k = 0; k<numberOfItems; k++){

        curr_pos = 0;

        // COPY HOST VARIABLES TO DEVICE  ------------------------------------------
        printf("Copying data from host to device..."); fflush(stdout);
        // startTime(&timer);

        copy_to_device(W_d, W_h, MAX);
        copy_to_device(P_d, P_h, MAX);
        copy_to_device(S_d, S_h, MAX);
        copy_to_device(U_d, U_h, MAX);
        printf("Copying Done for SOME DATA host to device...");
        copy_to_device(C_bar_d, &C_bar, 1); 
        copy_to_device(k_d, &k, 1);
        copy_to_device(q_d, &q_size, 1);

        
        cudaDeviceSynchronize();
        // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // LAUNCH KERNEL 1----------------------------------------------------------

        printf("Launching kernel 1..."); fflush(stdout);
        // startTime(&timer);

        const unsigned int THREADS_PER_BLOCK = 512;
        const unsigned int numBlocks = (q_size - 1)/THREADS_PER_BLOCK + 1;
        dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
        //INSERT CODE HERE to call kernel
        Kernel1<<<ceil(q_size/512.0), THREADS_PER_BLOCK>>>(W_d, P_d, S_d, U_d, k_d, q_d, P_star_d, W_star_d);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
        
        // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        copy_to_device(L_d, L_h, MAX);

        // LAUNCH KERNEL 2----------------------------------------------------------
        printf("Launching kernel 2..."); fflush(stdout);
        Kernel2<<<ceil(q_size/512.0), THREADS_PER_BLOCK>>>(W_d, P_d, S_d, L_d, U_d, k_d, q_d, W_star_d, P_star_d, Capacity, numberOfItems ); //int *w_hat, int *p_hat, int *s, int *L, int *U, int q, int h
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
        
        // Copy L_d to L_h and calculate L_bar; Then send L_bar to Kernel 3
        cuda_ret = cudaMemcpy(L_h, L_d, sizeof(float)*MAX, cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy L (lower bound) from device to host");

        int L_bar = -1;
        for (int i = 0; i<MAX; i++){
            L_bar = max(L_bar, L_h[i]); 
        }    
        printf("L_bar value is: = %d \n", L_bar);

        // LAUNCH KERNEL 3----------------------------------------------------------
        // q_d = 2*q_d;
        q_size = 2*q_size;
        int* Label_d;
        cuda_ret = cudaMalloc((void**) &Label_d, sizeof(int)*MAX);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        printf("Launching kernel 3..."); fflush(stdout);
        Kernel3<<<ceil(q_size/512.0), THREADS_PER_BLOCK>>>(L_bar, U_d, Label_d); //int *w_hat, int *p_hat, int *s, int *L, int *U, int q, int h
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
        cudaDeviceSynchronize();
        
        // Copy Label_d to Label_h 
        // int* Label_h;
        // cuda_ret = cudaMemcpy(Label_h, Label_d, sizeof(float)*MAX, cudaMemcpyDeviceToHost);
        // if(cuda_ret != cudaSuccess) FATAL("Unable to copy Label from device to host");

        // // Verify correctness -----------------------------------------------------

        // printf("Verifying results..."); fflush(stdout);

        // verify(A_h, B_h, C_h, n);

        // for (j = 0; j<q_size; j++){
            
        //     // iterate over all nodes in linked list
        //     if (node_list[j].ub > bnb_lb){
                                        
        //         node_list[q_size + curr_pos] = Kernel1(node_list[j]);
                
        //         if (node_list[q_size + curr_pos].ub > bnb_lb){
        //             curr_pos++;
        //         }
        //     }
        // }
        // q_size+= curr_pos;

        // if (q_size > MAX/2){
        //     printf("increasing q size");
        //     MAX *= 10;
        //     Node* node_list_new = (Node*)malloc(MAX*sizeof(Node));
        //     for(int q_i = 0; q_i < q_size; q_i ++){
        //         node_list_new[q_i] = node_list[q_i];
        //     }
        //     free(node_list);
        //     node_list = node_list_new;
        // }

        // if (q_size == 0){
        //     break;
        // }
    }

    
    
    printf(" Best val = %d, ", bnb_lb);

    end_t = clock();
    diff_t = (end_t - start_t)/(CLOCKS_PER_SEC/1000);

    printf("Execution time (ms) = %6.3f\n", diff_t);
    
    inst_time += diff_t;

    
    printf("\n");
    printf("Average time for instances of size %d = %0.00f \n", numberOfItems, inst_time/10.0);

 
    // Free memory -----------------------------------------------------------
    free(W_h);
    free(P_h);
    free(U_h);
    free(L_h);
    free(S_h);

    //INSERT CODE HERE to free device matrices
    cudaFree(W_d);
    cudaFree(P_d);
    cudaFree(U_d);
    cudaFree(L_d);
    cudaFree(S_d);
    cudaFree(C_bar_d);
    return 0;

}
        