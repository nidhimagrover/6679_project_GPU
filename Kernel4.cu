__global__ void Kernel4(int *w_hat, int *p_hat, int *s, int *U, int *Label, int *concatIndexList) {

    // Calculate global thread index based on the block and thread indices ----

    //INSERT KERNEL CODE HERE
    int l = blockDim.x*blockIdx.x + threadIdx.x;
    int j = concatIndexList[l];
    // Use global index to determine which elements to read, add, and write ---

    //How to define j?, Shouls we use Atomic Exchange? 
    if(Label[l] == 0)
    {
        atomicExch(& w_hat[l], w_hat[j]); 
        // w_hat[l] = w_hat[j]; // atomicExch()
        atomicExch(& p_hat[l], p_hat[j]);
        // p_hat[l] = p_hat[j];
        atomicExch(& s[l], s[j]);
        // s[l] = s[j];
        atomicExch(& U[l], U[j]);
        // U[l] = U[j];
    }
    __syncthreads();

}

