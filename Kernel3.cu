__global__ void Kernel3(int L_bar, int *U, int *Label) {
    // This Kernel is used to label nonpromising nodes

    // Use atomicMax to calculate L_bar
    // L_bar is maximum of all lower bounds.

    // Calculate global thread index based on the block and thread indices ----
    int e = blockDim.x*blockIdx.x + threadIdx.x; 

    // Use global index to determine which elements to read, add, and write ---
    if (e < q)
    {
        //INSERT KERNEL CODE HERE
        int Ue = U[e];
        
        if (Ue<=L_bar)
        {
            atomicExch(& Label[e], 0);
        }
        else
        {
            atomicExch(& Label[e],1);
        }
    }
    __syncthreads();

    // Transfer the Table "Label" (which has value 0/1 for each node) to CPU

}