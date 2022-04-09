__global__ void Kernel3(float *U, float *L_bar, int *Label) {
    // L_bar is maximum of all lower bounds.
    // Don't know if it should be int or float and pointer or not.

    // Calculate global thread index based on the block and thread indices ----
    int e = blockDim.x*blockIdx.x + threadIdx.x; // +q or +2q ?

    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE
    int Ue = U[e]
    
    if (Ue<=L_bar){
        Label[e]=0
    }
    else{
        Label[e]=1
    }
    __syncthreads();

}

