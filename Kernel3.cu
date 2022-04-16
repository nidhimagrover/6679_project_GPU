__global__ void Kernel3(int L_bar, int *U, int *Label) {
    // This Kernel is used to label nonpromising nodes

    // Use atomicMax to calculate L_bar
    // L_bar is maximum of all lower bounds.
    // Don't know if it should be int or float and pointer or not.


 
    // Calculate global thread index based on the block and thread indices ----
    int e = blockDim.x*blockIdx.x + threadIdx.x; // +q or +2q ?

    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE
    int Ue = U[e];
    
    if (Ue<=L_bar)
    {
        Label[e]=0;
    }
    else
    {
        Label[e]=1;
    }
    __syncthreads();

    //atomicExch?

    // Transfer the Table "Label" (which has value 0/1 for each node) to CPU

}