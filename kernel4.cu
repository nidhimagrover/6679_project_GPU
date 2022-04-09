__global__ void Kernel4(int *w_hat, int *p_hat, int *s, int *U, int *Label) {

    // Calculate global thread index based on the block and thread indices ----

    //INSERT KERNEL CODE HERE
    int l = blockDim.x*blockIdx.x + threadIdx.x;

    // Use global index to determine which elements to read, add, and write ---

    //How to define j?, Shouls we use Atomic Exchange? 
    if(Label[l] == 0){
        w_hat[l] = w_hat[j];
        p_hat[l] = p_hat[j];
        s[l] = s[j];
        U[l] = U[j];
    }
    __syncthreads();

}

