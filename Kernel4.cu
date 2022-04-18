__global__ void Kernel4(int *w_hat, int *p_hat, int *s, int *U, int *Label, int *concatIndexList, int *qq) {

    // Calculate global thread index based on the block and thread indices ----
    int q = *qq;
    //INSERT KERNEL CODE HERE
    int l = blockDim.x*blockIdx.x + threadIdx.x;
    int j = concatIndexList[l];
    // Use global index to determine which elements to read, add, and write ---

    if(Label[l] == 0 && l < q)
    {
        atomicExch(& w_hat[l], w_hat[j]); 
        atomicExch(& p_hat[l], p_hat[j]);
        atomicExch(& s[l], s[j]);
        atomicExch(& U[l], U[j]);
    }
    __syncthreads();

}

