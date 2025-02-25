#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <csr.h>
#include <time.h>


int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);
    
    struct CSR globalMat, localMat;
    int rank, size, global_n = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        globalMat = *readCSR_symmetric("cfd1.mtx");
        double left_bound, right_bound;
        gershgorin_bounds(&globalMat, &left_bound, &right_bound);
        printf("Eigenvalue bounds: [%.6f, %.6f]\n", left_bound, right_bound);
    }

    distributeCSR(rank == 0 ? &globalMat : NULL, &localMat, &global_n, 0, MPI_COMM_WORLD);
    // printLocalCSR(&localMat, rank, size, global_n, MPI_COMM_WORLD);

    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    double eigenvalue = distributedPowerIteration(&localMat, 150, global_n, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    double local_time = end_time - start_time;
    double exec_time = 0 ;
    MPI_Reduce(&local_time, &exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        printf("Execution time: %.6f seconds\n", exec_time);
    }


    // Use localMat for computation...
    
    // Cleanup
    free(localMat.rowPtr);
    free(localMat.colIdx);
    free(localMat.val);
    
    MPI_Finalize();
}

