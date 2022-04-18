# ISYE6679 project: CPU-GPU implementation

CPU-GPU implementation of Branch and Bound for 0/1 Knapsack problem

Instructions:

1) load the cuda module
2) compile
3) submit the job to PBS to run

### Basic usage:

1) Load the cuda module:

    ```
    prompt% module load cuda
    ```

2) Compile the example:

    ```
    prompt% make
    ```

    The Makefile contains many useful bits of information on how to compile a CUDA code

3) Submit the example to PBS:

    ```
    prompt% qsub run.pbs -q pace-ice-gpu
    ```


<!-- 4) Compare the program output. 

```
diff batch.err batch.err.ref
diff batch.log batch.log.ref
diff myoutput.log myoutput.log.ref
``` -->

