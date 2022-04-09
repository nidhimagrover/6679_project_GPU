__global__ void Kernel2(int *w_hat, int *p_hat, int *s, int *L, int *U, int q, int h) {

    // Calculate global thread index based on the block and thread indices ----

    //INSERT KERNEL CODE HERE
    int j = blockDim.x*blockIdx.x + threadIdx.x + q;
    int wj = w_hat[j], pj = p_hat[j], sj, = s[j];
    int i = h, wi, pi, what, phat = 0, psj, wsj;

    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE
    while (i<=n){
        wi = w[i];
        pi = p[i];
        /* Compute what, phat, wsj, psj, sj which are used to get the Upper bound Uj*/
        if (i ≥ sj){
            if (wj + wi ≤ c){
                w = wj + wi; 
                p = pj + pi; 
            }
            else if (phat == 0) {
            wsj = wi; 
            psj = pi;
            what = wj;
            phat = pj;
            sj = i;
            } 
        }
        if (__all(phat)){
            break;
        }
        __syncthreads();
        i = i+1;
    }
    for (;i<=n;i++){
        wi = w[i];
        pi = p[i];
        /* Compute of the lower bound Lj*/
        if (wj + wi <= c){
            wj = wj + wi;
            pj = pj + pi;
        }
    }
    /*Update of the tuple (w_hat_j,p_hat_j,s_j,U_j,L_j) in the global memory of the GPU*/
    AtomicExch(& w_hat[j], what); 
    AtomicExch(& p_hat[j ], phat); 
    AtomicExch(& s[j], sj );
    phat = phat + (c - what)*psj/wsj; 
    AtomicExch(& U[j],phat); 
    AtomicExch(& L[j], pj);
}


