__global__ void Kernel2(int *w_hat, int *p_hat, int *s, int *L, int *U, int* kk, int* qq, int *W_star, int *P_star, int Capacity, int numberOfItems) {

    // Calculate global thread index based on the block and thread indices ----
    int k = *kk;
    int q = *qq;
    //INSERT KERNEL CODE HERE
    int j = blockDim.x*blockIdx.x + threadIdx.x + q;
    int wj = w_hat[j], pj = p_hat[j], sj = s[j];
    int i = max(k, sj), wi, pi, what, phat = 0, psj, wsj;

    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE
    while (i < numberOfItems){
        wi = W_star[i];
        pi = P_star[i];
        /* Compute what, phat, wsj, psj, sj which are used to get the Upper bound Uj*/
        if (i >= sj)
        {
            if (wj + wi <= Capacity)
            {
                wj = wj + wi; 
                pj = pj + pi; 
            }
            else if (phat == 0) 
            {
                wsj = wi; 
                psj = pi;
                what = wj;
                phat = pj;
                sj = i;
            } 
        }
        if (__all_sync(__activemask(), phat))
        {
            break;
        }
        __syncthreads();
        i = i+1;
    }
    for (;i<=numberOfItems;i++)
    {
        wi = W_star[i];
        pi = P_star[i];
        /* Compute of the lower bound Lj*/
        if (wj + wi <= Capacity){
            wj = wj + wi;
            pj = pj + pi;
        }
    }
    // printf("Completed Calculating in Kernel 2");
    /*Update of the tuple (w_hat_j,p_hat_j,s_j,U_j,L_j) in the global memory of the GPU*/
    atomicExch(& w_hat[j], what); 
    atomicExch(& p_hat[j ], phat); 
    atomicExch(& s[j], sj );
    phat = phat + (Capacity - what)*psj/wsj; 
    atomicExch(& U[j],phat); 
    atomicExch(& L[j], pj);

    // printf("Completed Updating in Kernel 2");
}