__global__ void Kernel1(int *w_hat, int *p_hat, int *s, int *U, int k, int q) {

    // Calculate global thread index based on the block and thread indices ----

    //INSERT KERNEL CODE HERE
    int e = blockDim.x*blockIdx.x + threadIdx.x;

    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE
    int se = s[e], we = w_hat[e], pe = p_hat[e], Ue = U[e];
    if (k<se){
        we = we - w[k];
        pe = pe - p[k];
    }
    else{
        se = se + 1;
        Ue = 0;
    }
    AtomicExch(& w_hat[e+q], we);
    AtomicExch(& p_hat[e+q], pe);
    AtomicExch(& s[e+q], se);
    AtomicExch(& U[e], Ue);
}

