
#include <assert.h>

__global__ void Kernel1(int *w_hat, int *p_hat, int *s, int *U, int* kk, int* qq, int *P_star, int *W_star) 
{

    int k = *kk;
    int q = *qq;
    // Calculate global thread index based on the block and thread indices ----

    //INSERT KERNEL CODE HERE
    int e = blockDim.x*blockIdx.x + threadIdx.x;

    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE
    int se = s[e], we = w_hat[e], pe = p_hat[e], Ue = U[e];
    if (k<se)
    {
        we = we - W_star[k];
        pe = pe - P_star[k];
    }
    else{
        se = se + 1;
        Ue = 0;
    }
    // printf("Completed Calculating in Kernel 1");
    atomicExch(& w_hat[e+q], we);
    atomicExch(& p_hat[e+q], pe);
    atomicExch(& s[e+q], se);
    atomicExch(& U[e], Ue);

    // printf("Completed Updating in Kernel 1");
}