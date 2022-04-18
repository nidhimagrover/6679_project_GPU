#include <stdio.h>
#include <time.h>
#include "support.h"
#include "Kernel1.cu"
#include "Kernel2.cu"
#include "Kernel3.cu"
#include "Kernel4.cu"
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

int L_bar = -1; // Best Lower bound Initialization
int j, new_q_size;

// void print_stats(int itr, int* W_d){

// }
void *fixed_cudaMalloc(size_t len)
{
    void *p;
    if (cudaMalloc(&p, len) == cudaSuccess) return p;
    return 0;
}

void copy_to_device(int* d_arr, int* h_arr, int len){

    cudaError_t cuda_ret = cudaMemcpy(d_arr, h_arr, sizeof(int)*len, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) {
    // print(gpuErrchk( cudaMalloc(&d_arr, len*sizeof(int)) );
    printf("\nError %s\n", cudaGetErrorString(cuda_ret));
    FATAL("Unable to COPY memory to device");
    }
}

void copy_from_device_to_host(int* h_arr, int* d_arr, int len)
{
    cudaError_t cuda_ret = cudaMemcpy(h_arr, d_arr, sizeof(int)*len, cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) 
    {
    // print(gpuErrchk( cudaMalloc(&d_arr, len*sizeof(int)) );
        printf("\nError %s\n", cudaGetErrorString(cuda_ret)) ;
        FATAL("UNABLE to COPY memory back to host");
    }
}

int MAX = 100000;

int main(int argc, char**argv) {
    // creating file pointer to work with files
    FILE *fptr;
    FILE *fptr1;
    
    // // opening file in writing mode
    fptr = fopen("6679.txt", "w");

    fptr1 = fopen("200instance6679.txt", "w"); // Generating instance data for Gurobi

    clock_t start_t, end_t;
    double diff_t;
    
    cudaError_t cuda_ret;

    int maxVal = 100;
    
    srand(1);
    numberOfItems = 400;
    
    
    double inst_time = 0;
    // fprintf(fptr, "Testing instances of size = %d \n", numberOfItems);
    
    for (int it=1; it<=10; it++) // Loop to Generate 10 instances for 100 items
    {   
        printf("iterations = %d \n", it);
        int* W_star = (int*) malloc( numberOfItems*sizeof(int) ); // host
        int* P_star = (int*) malloc( numberOfItems*sizeof(int) ); // host

        int *W_star_d = (int*) fixed_cudaMalloc(sizeof(int)*numberOfItems); // device
        int *P_star_d = (int*) fixed_cudaMalloc(sizeof(int)*numberOfItems); // device

        Inst = getInstance(numberOfItems, maxVal); // Get Instance data
        // Read Data from Excel sheet obtained from Gurobi
        for (int iter = 0; iter < numberOfItems; iter++)
        {
            W_star[iter] = Inst.weight[iter];
            P_star[iter] = Inst.price[iter];
            
        }
        // for (int iter = 0; iter < numberOfItems; iter++)
        // {
        //     W_star[iter] = Inst.weight[iter];
        //     fprintf(fptr1, "%d ", W_star[iter]);
            
        // }
        // for (int iter = 0; iter < numberOfItems; iter++)
        // {
        //     P_star[iter] = Inst.price[iter];
        //     fprintf(fptr1, "%d ", P_star[iter]);
            
        // }
        Capacity = Inst.capacity;
        fprintf(fptr1, "%d ", Capacity);
        start_t = clock();
        
        // fprintf(fptr, "Setting up the problem...\n"); 
        // printf("Setting up the problem...\n"); 
        
        // ALLOCATING HOST MEMORY ----------------------------------------------
        int* W_h = (int*) malloc( MAX*sizeof(int) );
        int* P_h = (int*) malloc( MAX*sizeof(int) );
        int* S_h = (int*) malloc( MAX*sizeof(int) );
        int* U_h = (int*) malloc( MAX*sizeof(int) );
        int* L_h = (int*) malloc( MAX*sizeof(int) );
        int* Label_h = (int*) malloc( MAX*sizeof(int) );
        int* concatIndexList_h = (int*) malloc( MAX*sizeof(int) ); // for concatenating promising and non-promising nodes
        
        // int* C_bar_h = (int*) malloc( 1*sizeof(int) );
        // fprintf(fptr, "Allocating Host Memory...\n"); 
        // printf("Allocating Host Memory...\n"); 
        int w = 0, p = 0, s = -1;
        int C_bar;
        
        while (w <= Capacity){
            s++; // initial slack
            w += W_star[s];
            p += P_star[s];
        }

        // ATTRIBUTES OF ROOT NODE: Initialize host variables ----------------------------------------------
        S_h[0] = s;
        P_h[0] = (p - P_star[s]);
        W_h[0] = (w - W_star[s]);
        C_bar = Capacity - (w - W_star[s]);
        U_h[0] = (p - P_star[s]) + (C_bar * P_star[s]/(double)W_star[s]);

        int wx = 0;
        L_bar = (p - P_star[s]);
        for (int i = s+1; i < numberOfItems; i++){
            if (wx + W_star[i] <= C_bar){
                wx += W_star[i];
                L_bar += P_star[i];
            }
        }
        L_h[0] = L_bar;
        // fprintf(fptr, "Initialization of Attributes of Root Node...\n"); 

        copy_to_device(W_star_d, W_star, numberOfItems);
        copy_to_device(P_star_d, P_star, numberOfItems);

        // Allocate device variables ----------------------------------------------

        q_size = 1; // initialized to 1 because there is only one node i.e. root node
        //new_q_size = q_size;
        
        int *W_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX);
        int *P_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX);
        int *S_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX);
        int *U_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX);
        int *L_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX);
        int *C_bar_d = (int*) fixed_cudaMalloc(sizeof(int));
        int *k_d = (int*) fixed_cudaMalloc(sizeof(int));
        int *q_d = (int*) fixed_cudaMalloc(sizeof(int));
        int *Label_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX);
        int *concatIndexList_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX);
        cudaDeviceSynchronize();
        // fprintf(fptr, "Device variables Allocated ...\n"); 
        
        copy_to_device(W_d, W_h, q_size);
        copy_to_device(P_d, P_h, q_size);
        copy_to_device(S_d, S_h, q_size);
        copy_to_device(U_d, U_h, q_size);
        copy_to_device(C_bar_d, &C_bar, 1); // Residual Capacity to device
        copy_to_device(L_d, L_h, q_size);
        cudaDeviceSynchronize();
        
        for (int k = 0; k<numberOfItems; k++) // numberOfItems
        {   
            // if (k%10 == 0){
            //     print_stats(k, W_d);
            // }
            copy_to_device(k_d, &k, 1);
            copy_to_device(q_d, &q_size, 1);
        
            cudaDeviceSynchronize();
            // COPY HOST VARIABLES TO DEVICE  ------------------------------------------
            // fprintf(fptr, "Copying data from host to device..."); 
            // printf("iteration for item %d -------------------------\n", k);
            
            
            // fprintf(fptr, "Launching kernel 1 for Item:  %d ...\n", k); 

            const unsigned int THREADS_PER_BLOCK = 512;
            const unsigned int numBlocks = (q_size - 1)/THREADS_PER_BLOCK + 1;
            dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);

            
            Kernel1<<<ceil(q_size/512.0), THREADS_PER_BLOCK>>>(W_d, P_d, S_d, U_d, k_d, q_d, P_star_d, W_star_d); // This kernel operates on q nodes to create another q nodes. Therefore total 2q nodes
            cuda_ret = cudaDeviceSynchronize();
            if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
            
            
            // fprintf(fptr, "Launching kernel 2 for Item:  %d ...\n", k); 
            Kernel2<<<ceil(q_size/512.0), THREADS_PER_BLOCK>>>(W_d, P_d, S_d, L_d, U_d, k_d, q_d, W_star_d, P_star_d, Capacity, numberOfItems ); //int *w_hat, int *p_hat, int *s, int *L, int *U, int q, int h
            cuda_ret = cudaDeviceSynchronize();
            if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
            
            // Copy L_d to L_h and calculate L_bar; Then send L_bar to Kernel 3
            // printf("Value of Lower Qsize: %d and iteration item: %d\n", q_size,k);
            copy_from_device_to_host(L_h, L_d, (2*q_size));
            // printf("After Value of Lower Qsize: %d and iteration item: %d\n", 2*q_size,k);
            for (int i = q_size; i<2*q_size; i++) // Function atomicMax(): shoud have been implemented in GPU
            {
                // fprintf(fptr, "L_bar value is: = %d, %d\n", L_bar, L_h[i]);
                L_bar = max(L_bar, L_h[i]);
            }    
            // Segmentation Fault Core dumped
            // fprintf(fptr, "L_bar value is: = %d\n", L_bar, L_h[0]);

             
            // q_d = 2*q_d;
            q_size = 2*q_size;
            // fprintf(fptr, "Q_size 1st: %d in Item: %d \n", q_size, k);
            // fprintf(fptr, "Launching kernel 3 for Item:  %d ...\n", k); 
            
            // copy_from_device_to_host(U_h, U_d, q_size);
            
            // printf("Value of item (i.e. k from for loop): %d and Qsize: %d\n", k, q_size);
            // for (int kkk = 0; kkk < q_size; kkk++)
            // {
            //     printf("Node: %d   UB: %d  LB: %d  LBar: %d \n", kkk, U_h[kkk], L_h[kkk], L_bar);
            // }
            
            Kernel3<<<ceil(q_size/512.0), THREADS_PER_BLOCK>>>(L_bar, U_d, Label_d); //int *w_hat, int *p_hat, int *s, int *L, int *U, int q, int h
            cuda_ret = cudaDeviceSynchronize();
            if(cuda_ret != cudaSuccess) {printf("\nError %s\n", cudaGetErrorString(cuda_ret)); FATAL("Unable to launch kernel");}
            cudaDeviceSynchronize();
            
            // Copy Label_d to Label_h 
            // printf("Value of Upper Qsize: %d\n", q_size);
            copy_from_device_to_host(Label_h, Label_d, q_size); //q_size has been doubled
            
            // Code to Determine Concatenation Indices
            int left = 0;
            int right = q_size-1;
            while(left < right)
            {
                if (Label_h[left] == 0) // && ())
                {   
                    if (Label_h[right] == 1)
                    {
                        concatIndexList_h[left] = right;
                        concatIndexList_h[right] = left;
                        left = left + 1;
                        right = right - 1;
                    }
                    else
                    {
                        concatIndexList_h[right] = right;
                        right = right - 1;
                    }

                }
                else
                {
                    concatIndexList_h[left] = left;
                    left = left + 1;
                    if (Label_h[right] == 0)
                    {
                        concatIndexList_h[right] = right;
                        right = right - 1;
                    }
                }
                
            }
            if (left == right)
                {
                    concatIndexList_h[left] = left;
                }
            
            // fprintf(fptr, "Launching kernel 4 for Item:  %d ...\n", k); 
            copy_to_device(concatIndexList_d, concatIndexList_h, MAX);
            // copy_to_device(Label_d, Label_h, MAX);
            
            // printf("Before kernel 4: %d and Qsize: %d\n", k, q_size);
            // for (int kkk = 0; kkk < q_size; kkk++)
            // {
            //     printf("Node: %d  label: %d concat_idx: %d  \n", kkk, Label_h[kkk], concatIndexList_h[kkk]);
            // }

            copy_to_device(q_d, &q_size, 1);
            // printf("Copying q_size to device:  %d ...\n", q_size); 
            Kernel4<<<ceil(q_size/512.0), THREADS_PER_BLOCK>>>(W_d, P_d, S_d, U_d, Label_d, concatIndexList_d, q_d);
            cuda_ret = cudaDeviceSynchronize();
            if(cuda_ret != cudaSuccess){printf("\nError %s\n", cudaGetErrorString(cuda_ret)); FATAL("Unable to Launch Kernel 4");}
            cudaDeviceSynchronize();
            
            // copy_from_device_to_host(U_h, U_d, q_size);

            // printf("After kernel 4: %d and Qsize: %d\n", k, q_size);
            // for (int kkk = 0; kkk < q_size; kkk++)
            // {
            //     printf("Node: %d   UB: %d  LB: %d  LBar: %d \n", kkk, U_h[kkk], L_h[kkk], L_bar);
            // }

            // Finally update q_size = q_size (which is actually twice its earlier size) - number of non-promising nodes
            int count = 0;
            for (int i = 0; i<q_size; i++)
            {
                if (Label_h[i] == 1){
                    count = count + 1;
                }
            }
            q_size = count; // updating q_size after removing non-promising nodes
            // printf("Q_size is: %d and Max is: %d \n", q_size,MAX);
            if (q_size > MAX/2 - 1)
            {
                // printf("Q_size enter if: %d\n", q_size);
                
                copy_from_device_to_host(W_h, W_d, q_size);
                copy_from_device_to_host(P_h, P_d, q_size);
                copy_from_device_to_host(S_h, S_d, q_size);
                copy_from_device_to_host(U_h, U_d, q_size);
                copy_from_device_to_host(L_h, L_d, q_size);
                // copy_from_device_to_host(Label_h, Label_d, q_size);
                // copy_from_device_to_host(concatIndexList_h, concatIndexList_d, q_size);
                cudaDeviceSynchronize();
                // size_t bbb = sizeof(W_d);
                // printf("Size of W_d before: %d\n", bbb);

                cudaFree(W_d);
                cudaFree(P_d);
                cudaFree(S_d);
                cudaFree(U_d);
                cudaFree(L_d);
                cudaFree(Label_d);
                cudaFree(concatIndexList_d);

                MAX *= 2;
                
                
                W_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX); // device
                P_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX); // device
                S_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX); // device
                U_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX); // device
                L_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX); // device
                Label_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX); // device
                concatIndexList_d = (int*) fixed_cudaMalloc(sizeof(int)*MAX); // device
                // size_t bbb1 = sizeof(W_d);
                // printf("Size of W_d after: %d\n", bbb1);
                cudaDeviceSynchronize();

                copy_to_device(W_d, W_h, q_size);
                cudaDeviceSynchronize();
                copy_to_device(P_d, P_h, q_size);
                cudaDeviceSynchronize();
                copy_to_device(S_d, S_h, q_size);
                cudaDeviceSynchronize();
                copy_to_device(U_d, U_h, q_size);
                cudaDeviceSynchronize();
                copy_to_device(L_d, L_h, q_size);
                // copy_to_device(Label_d, Label_h, q_size);
                // copy_to_device(concatIndexList_d, concatIndexList_h, q_size);
                
                cudaDeviceSynchronize();
                
                free(W_h);
                free(P_h);
                free(S_h);
                free(U_h);
                free(L_h);
                free(Label_h);
                free(concatIndexList_h);
                W_h = (int*) malloc( MAX*sizeof(int) );
                P_h = (int*) malloc( MAX*sizeof(int) );
                S_h = (int*) malloc( MAX*sizeof(int) );
                U_h = (int*) malloc( MAX*sizeof(int) );
                L_h = (int*) malloc( MAX*sizeof(int) );
                Label_h = (int*) malloc( MAX*sizeof(int) );
                concatIndexList_h = (int*) malloc( MAX*sizeof(int) );
                
                cudaDeviceSynchronize();
                // free(node_list);
            }

            if (q_size == 0)
            {
                break;
            }
        }

        
        
        fprintf(fptr, " Best val = %d, ", L_bar);
        fprintf(fptr1, "%d\n", L_bar);
        end_t = clock();
        diff_t = (end_t - start_t)/(CLOCKS_PER_SEC/1000);

        fprintf(fptr, "Execution time (ms) = %6.3f\n", diff_t);
        
        inst_time += diff_t;

        free(W_h);
        free(P_h);
        free(S_h);
        free(U_h);
        free(L_h);
        free(Label_h);
        free(W_star);
        free(P_star);
        free(concatIndexList_h);

        //INSERT CODE HERE to free device matrices
        cudaFree(W_d);
        cudaFree(P_d);
        cudaFree(S_d);
        cudaFree(U_d);
        cudaFree(L_d);
        cudaFree(C_bar_d);
        cudaFree(k_d);
        cudaFree(q_d);
        cudaFree(Label_d);
        cudaFree(W_star_d);
        cudaFree(P_star_d);
        cudaFree(concatIndexList_d);

    }
    fprintf(fptr, "\n");
    fprintf(fptr, "Average time for instances of size %d = %0.00f \n", numberOfItems, inst_time/10.0);

    fclose(fptr);
    fclose(fptr1);
 
    // Free memory -----------------------------------------------------------
    
    return 0;

}
        