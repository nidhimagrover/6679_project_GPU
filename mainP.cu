#include <stdio.h>
#include "support.h"
#include <time.h>
#include "Kernel1.cu"
#include "Kernel2.cu"
#include "Kernel3.cu"
#include "generate_instance.h"
#ifndef max
    #define max(a,b) ((a) > (b) ? (a) : (b))
#endif

typedef struct {
    unsigned int w_hat, p_hat, ub, lb, s;
} Node;

struct Instance Inst;
unsigned short int numberOfItems;

unsigned int bnb_lb;
unsigned short int k, j, q_size, new_q_size;
int main(int argc, char**argv) {

    clock_t start_t, end_t;
    double diff_t;
    
    Timer timer;
    cudaError_t cuda_ret;

    int maxVal = 1000;
    unsigned int MAX = 100000;
    srand(1);

    for (numberOfItems=1000; numberOfItems<=10000; numberOfItems=numberOfItems+1000) // Loop to Generate Instances and Test over DIfferent set of Items
    {   
        double inst_time = 0;
        printf("Testing instances of size = %d \n", numberOfItems);
    
        for (int iter=1; iter<=10; iter++) // Loop to Generate 10 instances for a given set of items
        {   
            printf("iterations = %d, ", iter);
            
            Inst = getInstance(numberOfItems, maxVal); // Get Instance data
            start_t = clock();

            Node* node_list = (Node*)malloc(MAX*sizeof(Node)); // Allocating space for a pre-defined set of nodes
            
            printf("\nSetting up the problem..."); fflush(stdout);
            startTime(&timer);

            unsigned int n;
            n = numberOfItems;
        
            int* W_h = (int*) malloc( MAX*sizeof(int) );
            int* P_h = (int*) malloc( MAX*sizeof(int) );
            int* S_h = (int*) malloc( MAX*sizeof(int) );
            int* U_h = (int*) malloc( MAX*sizeof(int) );
            int* L_h = (int*) malloc( MAX*sizeof(int) );
            // int* C_bar_h = (int*) malloc( 1*sizeof(int) );
            int* k = (int*) malloc( 1*sizeof(int) );

            int w = 0, p = 0, s = -1;
            int C_bar;
            
            while (w <= Inst.capacity){
                s++;
                w += Inst.weight[s];
                p += Inst.price[s];
            }
            // Attributes of Root Node
            // Initialize host variables ----------------------------------------------
            node_list[0].s = s; // Slack variable
            S_h[0] = s;
            node_list[0].p_hat = (p - Inst.price[s]);
            P_h[0] = (p - Inst.price[s]);
            node_list[0].w_hat = (w - Inst.weight[s]);
            W_h[0] = (w - Inst.weight[s]);
            C_bar = Inst.capacity - node_list[0].w_hat;
            node_list[0].ub = node_list[0].p_hat + (C_bar * Inst.price[s]/(double)Inst.weight[s]);
            U_h[0] = node_list[0].p_hat + (C_bar * Inst.price[s]/(double)Inst.weight[s]);

            int wx = 0;
            bnb_lb = node_list[0].p_hat;
            for (int i = s+1; i < numberOfItems; i++){
                if (wx + Inst.weight[i] <= C_bar){
                    wx += Inst.weight[i];
                    bnb_lb += Inst.price[i];
                }
            }
            node_list[0].lb = bnb_lb;
            L_h[0] = bnb_lb;
            
            stopTime(&timer); printf("%f s\n", elapsedTime(timer));
            printf("    Number of Items = %u\n", numberOfItems);

            // Allocate device variables ----------------------------------------------

            printf("Allocating device variables..."); fflush(stdout);
            startTime(&timer);

            q_size = 1; // initialized to 1 because there is only one node i.e. root node
            //new_q_size = q_size;
            int curr_pos = 0;

            int* W_d;
            cuda_ret = cudaMalloc((void**) &W_d, sizeof(int)*q_size);
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

            int* P_d;
            cuda_ret = cudaMalloc((void**) &P_d, sizeof(int)*q_size);
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

            int* S_d;
            cuda_ret = cudaMalloc((void**) &S_d, sizeof(int)*q_size);
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

            int* U_d;
            cuda_ret = cudaMalloc((void**) &U_d, sizeof(int)*q_size);
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

            int* L_d;
            cuda_ret = cudaMalloc((void**) &L_d, sizeof(int)*q_size);
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

            int* C_bar_d;
            cuda_ret = cudaMalloc((void**) &C_bar_d, sizeof(int)*1);
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

            int* k_d;
            cuda_ret = cudaMalloc((void**) &k_d, sizeof(int)*1);
            if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

            cudaDeviceSynchronize();
            stopTime(&timer); printf("%f s\n", elapsedTime(timer));

            for (k = 0; k<numberOfItems; k++){

                curr_pos = 0;

                // Copy host variables to device ------------------------------------------

                printf("Copying data from host to device..."); fflush(stdout);
                startTime(&timer);

                cuda_ret = cudaMemcpy(W_d, W_h, sizeof(int)*q_size, cudaMemcpyHostToDevice);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

                cuda_ret = cudaMemcpy(P_d, P_h, sizeof(int)*q_size, cudaMemcpyHostToDevice);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

                cuda_ret = cudaMemcpy(S_d, S_h, sizeof(int)*q_size, cudaMemcpyHostToDevice);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

                cuda_ret = cudaMemcpy(U_d, U_h, sizeof(int)*q_size, cudaMemcpyHostToDevice);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

                cuda_ret = cudaMemcpy(C_bar_d, C_bar, sizeof(int)*1, cudaMemcpyHostToDevice);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

                cuda_ret = cudaMemcpy(k_d, k, sizeof(int)*1, cudaMemcpyHostToDevice);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

                cudaDeviceSynchronize();
                stopTime(&timer); printf("%f s\n", elapsedTime(timer));

                // LAUNCH KERNEL 1----------------------------------------------------------

                printf("Launching kernel..."); fflush(stdout);
                startTime(&timer);

                const unsigned int THREADS_PER_BLOCK = 512;
                const unsigned int numBlocks = (q_size - 1)/THREADS_PER_BLOCK + 1;
                dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
                //INSERT CODE HERE to call kernel
                Kernel1<<<ceil(q_size/512.0), THREADS_PER_BLOCK>>>(W_d, P_d, S_d, U_d, k_d, q_size);
                cuda_ret = cudaDeviceSynchronize();
                if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
                stopTime(&timer); printf("%f s\n", elapsedTime(timer));

                // Copy device variables to host ----------------------------------------

                printf("Copying data from device to host..."); fflush(stdout);
                startTime(&timer);
                
                // // LAUNCH KERNEL 2----------------------------------------------------------
                // int i = max(k, se);
                // printf("Launching kernel..."); fflush(stdout);
                // startTime(&timer);

                // const unsigned int THREADS_PER_BLOCK = 512;
                // const unsigned int numBlocks = (q_size - 1)/THREADS_PER_BLOCK + 1;
                // dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
                // //INSERT CODE HERE to call kernel
                // Kernel1<<<ceil(q_size/512.0), THREADS_PER_BLOCK>>>(W_d, P_d, S_d, U_d, k_d, q_size);
                // cuda_ret = cudaDeviceSynchronize();
                // if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
                // stopTime(&timer); printf("%f s\n", elapsedTime(timer));




                //INSERT CODE HERE to copyfrom device to host
                cuda_ret = cudaMemcpy(W_h, W_d, sizeof(int)*q_size, cudaMemcpyDeviceToHost);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy W from device to host");

                cuda_ret = cudaMemcpy(P_h, P_d, sizeof(int)*q_size, cudaMemcpyDeviceToHost);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy P from device to host");

                cuda_ret = cudaMemcpy(S_h, S_d, sizeof(int)*q_size, cudaMemcpyDeviceToHost);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy S from device to host");

                cuda_ret = cudaMemcpy(U_h, U_d, sizeof(int)*q_size, cudaMemcpyDeviceToHost);
                if(cuda_ret != cudaSuccess) FATAL("Unable to copy U from device to host");

                cudaDeviceSynchronize();
                stopTime(&timer); printf("%f s\n", elapsedTime(timer));

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

            // free(node_list);
            
            printf(" Best val = %d, ", bnb_lb);

            end_t = clock();
            diff_t = (end_t - start_t)/(CLOCKS_PER_SEC/1000);

            printf("Execution time (ms) = %6.3f\n", diff_t);
            
            inst_time += diff_t;
        
        }
        
        printf("\n");
        printf("Average time for instances of size %d = %0.00f \n", numberOfItems, inst_time/10.0);

    }
    // Free memory -----------------------------------------------------------
    free(W_h);
    free(P_h);
    free(U_h);
    free(L_h);
    free(S_h);
    free(k);
    //INSERT CODE HERE to free device matrices
    cudaFree(W_d);
    cudaFree(P_d);
    cudaFree(U_d);
    cudaFree(L_d);
    cudaFree(S_d);
    cudaFree(k);
    cudaFree(C_bar_d)
    return 0;

}
        