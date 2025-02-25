#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cblas.h>
#include <mpi.h>
#pragma once

struct CSR {
    int n;
    int nnz;
    int *rowPtr;
    int *colIdx;
    double *val;
};

void* build_csr(struct CSR *csr, int n, int nnz, int *rowPtr, int *colIdx, double *val) {
    csr->n = n;
    csr->nnz = nnz;
    csr->rowPtr = rowPtr;
    csr->colIdx = colIdx;
    csr->val = val;
}

void free_csr(struct CSR *csr) {
    free(csr->rowPtr);
    free(csr->colIdx);
    free(csr->val);
    free(csr);
}


struct CSR* readCSR(const char *filename, struct CSR *csr) {

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL ;
    }

    char line[128];
    while(fgets(line, 128, file)[0] == '%');
    int n, nnz; 
    sscanf(line, "%d %d %d", &n, &n, &nnz);
    printf("n: %d, nnz: %d\n", n, nnz);

    int *rowPtr = (int*)malloc((n + 1) * sizeof(int));
    int *colIdx = (int*)malloc(nnz * sizeof(int));
    double *val = (double*)malloc(nnz * sizeof(double));
    
    int i = 0 ;     
    int j = 1 ;    
    int row, col ;
    double value ;

    for (i = 0; i < nnz; i++){
        fscanf(file, "%d %d %lf", &row, &col, &value);
        row--;
        col--;
        rowPtr[row + 1]++;
        colIdx[i] = col;
        val[i] = value;
    }
    fclose(file);

    for (int i = 0; i < n; i++) {
        rowPtr[i + 1] += rowPtr[i];
    }

    if(!csr){
        struct CSR *csr2 = (struct CSR*)malloc(sizeof(struct CSR));
        build_csr(csr2, n, nnz, rowPtr, colIdx, val);
        return csr2;
    }
    else{
        build_csr(csr, n, nnz, rowPtr, colIdx, val);
        return csr;
    }
}


struct CSR *readCSR_symmetric(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Error opening file");
        return NULL;
    }

    char line[128];
    while(fgets(line, 128, f)[0] == '%');

    // Read matrix dimensions: nrows, ncols, nnz
    int nrows, ncols, nnz;
    if (sscanf(line, "%d %d %d", &nrows, &ncols, &nnz) != 3) {
        fprintf(stderr, "Failed to read matrix size.\n");
        fclose(f);
        return NULL;
    }

    // Allocate temporary storage for the triplets.
    // For symmetric matrices, off-diagonals will be stored twice.
    int max_entries = nnz * 2;
    int *I = (int *)malloc(max_entries * sizeof(int));
    int *J = (int *)malloc(max_entries * sizeof(int));
    double *V = (double *)malloc(max_entries * sizeof(double));

    int count_entries = 0;
    for (int k = 0; k < nnz; k++) {
        int i, j;
        double value;
        if (fscanf(f, "%d %d %lf", &i, &j, &value) != 3) {
            fprintf(stderr, "Error reading matrix entries.\n");
            free(I); free(J); free(V);
            fclose(f);
            return NULL;
        }
        // Convert from 1-based to 0-based indexing.
        i--; j--;
        I[count_entries] = i;
        J[count_entries] = j;
        V[count_entries] = value;
        count_entries++;

        // For symmetric matrices, if off-diagonal, add the symmetric element.
        // if (symmetric && i != j) {
        if (i != j){
            I[count_entries] = j;
            J[count_entries] = i;
            V[count_entries] = value;
            count_entries++;
        }
    }
    fclose(f);
    int actual_nnz = count_entries;

    // Allocate CSR arrays.
    int *rowPtr = (int *)malloc((nrows + 1) * sizeof(int));
    int *colIdx = (int *)malloc(actual_nnz * sizeof(int));
    double *vals = (double *)malloc(actual_nnz * sizeof(double));

    // Count the number of nonzero entries per row.
    memset(rowPtr, 0, (nrows + 1) * sizeof(int));
    for (int k = 0; k < actual_nnz; k++) {
        rowPtr[I[k] + 1]++;
    }
    // Cumulative sum to get the row pointer.
    for (int i = 0; i < nrows; i++) {
        rowPtr[i + 1] += rowPtr[i];
    }

    // Temporary copy of rowPtr for use in placing elements.
    int *temp_rowPtr = (int *)malloc((nrows + 1) * sizeof(int));
    memcpy(temp_rowPtr, rowPtr, (nrows + 1) * sizeof(int));
    // Place the triplets into the CSR structure.
    for (int k = 0; k < actual_nnz; k++) {
        int i = I[k];
        int dest = temp_rowPtr[i];
        colIdx[dest] = J[k];
        vals[dest] = V[k];
        temp_rowPtr[i]++;
    }
    free(temp_rowPtr);
    free(I);
    free(J);
    free(V);

    // Allocate and fill the CSR struct.
    struct CSR *A = (struct CSR *)malloc(sizeof(struct CSR));
    A->n = nrows;
    A->nnz = actual_nnz;
    A->rowPtr = rowPtr;
    A->colIdx = colIdx;
    A->val = vals;

    return A;
}


void matrix_vector_csr(struct CSR *csr, double *x, double *y) {
    int n = csr->n;
    int *rowPtr = csr->rowPtr;
    int *colIdx = csr->colIdx;
    double *val = csr->val;
    
    int i, j;
    for (i = 0; i < n; i++) {
        y[i] = 0.0;
        for (j = rowPtr[i]; j < rowPtr[i+1]; j++) {
            y[i] += val[j] * x[colIdx[j]];
        }
        // printf("y[%d] = %f\n", i, y[i]);
    }

}

double  vector_norm(double *x, int n) {
    double norm = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        norm += x[i] * x[i];
    }

    return sqrt(norm);
}

double calculateError(struct CSR *csr, double *x, double *b, int n) {
    double error = 0.0;
    int i;

    double *res = (double*)malloc(n * sizeof(double));

    matrix_vector_csr(csr, x, res);
    for (i = 0; i < n; i++) {
            res[i] = b[i] - res[i];
        }

    error = cblas_dnrm2(n, res, 1) / cblas_dnrm2(n, b, 1);
    free(res);

    return error;
}

void gershgorin_bounds(struct CSR *csr, double *left_bound, double *right_bound) {
    int n = csr->n;
    double min_bound = __FLT64_MAX__;
    double max_bound = -__FLT64_MAX__;

    for (int i = 0; i < n; i++) {
        double diag = 0.0;
        double radius = 0.0;
        // Loop over the nonzeros in row i.
        for (int k = csr->rowPtr[i]; k < csr->rowPtr[i + 1]; k++) {
            int j = csr->colIdx[k];
            double aij = csr->val[k];
            if (j == i) {
                diag = aij;
            } else {
                radius += fabs(aij);
            }
        }
        double lower = diag - radius;
        double upper = diag + radius;
        if (lower < min_bound)
            min_bound = lower;
        if (upper > max_bound)
            max_bound = upper;
    }

    *left_bound = min_bound;
    *right_bound = max_bound;
}

double powerIteration(struct CSR* csr, double* x, int n_iter){

    int n = csr->n ;
    double* x1 = (double*)malloc(n * sizeof(double));
    double* x2 = (double*)malloc(n * sizeof(double));
    double* _ = NULL ;

    for (int i = 0; i < n; i++) {
        x[i] = (double)rand() / RAND_MAX;
    }

    double inv_norm, lambda ;
    for (int i = 0 ; i < n_iter ; i++){

        inv_norm = 1.0/cblas_dnrm2(n, x, 1); // Calculate norm2
        cblas_dscal(n, inv_norm, x, 1);  // Normalise
        matrix_vector_csr(csr, x, x1) ; // Normalise
        _ = x;
        x = x1;
        x1 = _;

        // lambda = cblas_ddot(n, x, 1, x1, 1) ; // Calculate dot product
        // printf("Eigenvalue: %f\n", lambda);
        // memcpy(x, x1, n * sizeof(double)) ;
    }

    lambda = cblas_ddot(n, x, 1, x1, 1) ; // Calculate dot product
    // printf("Eigenvalue: %f\n", lambda);
    
    // free(x1) ;
    free(x2) ;
    free(x1) ;

    return lambda;
}


void distributeCSR(const struct CSR *globalMat, struct CSR *localMat, int *global_n, int root, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) *global_n = globalMat->n;
    MPI_Bcast(global_n, 1, MPI_INT, root, comm);


    if (rank == root) {
        // Root process splits and sends data
        int n = globalMat->n;
        int *rowPtr = globalMat->rowPtr;
        int *colIdx = globalMat->colIdx;
        double *val = globalMat->val;

        for (int r = 0; r < size; r++) {
            // Calculate row range for process r
            int quotient = n / size;
            int remainder = n % size;
            int start_row = (r < remainder) ? r * (quotient + 1) : remainder * (quotient + 1) + (r - remainder) * quotient;
            int end_row = (r < remainder) ? start_row + (quotient + 1) : start_row + quotient;

            // Extract local CSR data for process r
            int local_n = end_row - start_row;
            int start_idx = rowPtr[start_row];
            int end_idx = rowPtr[end_row];
            int local_nnz = end_idx - start_idx;

            // Prepare local_rowPtr (offset-adjusted)
            int *local_rowPtr = malloc((local_n + 1) * sizeof(int));
            for (int i = 0; i <= local_n; i++) {
                local_rowPtr[i] = rowPtr[start_row + i] - start_idx;
            }

            if (r == root) {
                // Root keeps its own copy
                localMat->n = local_n;
                localMat->nnz = local_nnz;
                localMat->rowPtr = local_rowPtr;
                
                localMat->colIdx = malloc(local_nnz * sizeof(int));
                memcpy(localMat->colIdx, &colIdx[start_idx], local_nnz * sizeof(int));
                
                localMat->val = malloc(local_nnz * sizeof(double));
                memcpy(localMat->val, &val[start_idx], local_nnz * sizeof(double));
            } else {
                // Send metadata
                MPI_Send(&local_n, 1, MPI_INT, r, 0, comm);
                MPI_Send(&local_nnz, 1, MPI_INT, r, 1, comm);
                
                // Send rowPtr, colIdx, and val
                MPI_Send(local_rowPtr, local_n + 1, MPI_INT, r, 2, comm);
                MPI_Send(&colIdx[start_idx], local_nnz, MPI_INT, r, 3, comm);
                MPI_Send(&val[start_idx], local_nnz, MPI_DOUBLE, r, 4, comm);
                
                free(local_rowPtr);
            }
        }
    } else {
        // Receive data from root
        MPI_Recv(&localMat->n, 1, MPI_INT, root, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&localMat->nnz, 1, MPI_INT, root, 1, comm, MPI_STATUS_IGNORE);
        
        int local_n = localMat->n;
        int local_nnz = localMat->nnz;

        // Receive rowPtr
        localMat->rowPtr = malloc((local_n + 1) * sizeof(int));
        MPI_Recv(localMat->rowPtr, local_n + 1, MPI_INT, root, 2, comm, MPI_STATUS_IGNORE);

        // Receive colIdx and val
        if (local_nnz > 0) {
            localMat->colIdx = malloc(local_nnz * sizeof(int));
            MPI_Recv(localMat->colIdx, local_nnz, MPI_INT, root, 3, comm, MPI_STATUS_IGNORE);
            
            localMat->val = malloc(local_nnz * sizeof(double));
            MPI_Recv(localMat->val, local_nnz, MPI_DOUBLE, root, 4, comm, MPI_STATUS_IGNORE);
        } else {
            localMat->colIdx = NULL;
            localMat->val = NULL;
        }
    }
}

void printLocalCSR(const struct CSR *localMat, int rank, int size, int global_n, MPI_Comm comm) {
    // Calculate global row offset for this process
    int quotient = global_n / size;
    int remainder = global_n % size;
    int start_row = (rank < remainder) ? rank * (quotient + 1) : remainder * (quotient + 1) + (rank - remainder) * quotient;

    // Print in rank order to avoid output interleaving
    for (int target_rank = 0; target_rank < size; target_rank++) {
        MPI_Barrier(comm);
        if (rank == target_rank) {
            printf("\nProcess %d local matrix (handling global rows %d-%d):\n",
                   rank, start_row, start_row + localMat->n - 1);
            
            if (localMat->n == 0) {
                printf("  No local rows\n");
            } else {
                for (int i = 0; i < localMat->n; i++) {
                    printf("  Local row %2d (global %2d): ", i, start_row + i);
                    for (int j = localMat->rowPtr[i]; j < localMat->rowPtr[i+1]; j++) {
                        printf("(%d, %.2f) ", localMat->colIdx[j], localMat->val[j]);
                    }
                    printf("\n");
                }
            }
            fflush(stdout);  // Ensure output appears immediately
        }
    }
    MPI_Barrier(comm);
    if (rank == 0) printf("\n");
}


double distributedPowerIteration(struct CSR *csr, int max_iter, int global_n, MPI_Comm comm) {
    int rank, size;
    int n = csr->n ;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    
    int *recvcounts = (int*)malloc(size * sizeof(int));
    int *displs     = (int*)malloc(size * sizeof(int));
    MPI_Allgather(&n, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    
    double inv_norm = 0.0 ;
    double lambda_local = 0.0 ;
    double lambda = 0.0 ;
    double local_sum = 0.0 ; 
    double global_sum = 0.0 ;
    
    
    int offset = displs[rank] ;
    double* x = (double*)malloc(global_n * sizeof(double));
    if (rank == 0) {
        for (int i = 0; i < global_n; i++) {
            x[i] = (double)rand() / RAND_MAX;
        }
    }
    MPI_Bcast(x, global_n, MPI_DOUBLE, 0, comm);


    double *x_new = (double*)malloc(global_n * sizeof(double));
    double *x_local = (double*)malloc(n * sizeof(double));
    double* _ = NULL ;
    memset(x_new, 0, global_n * sizeof(double));

    for (int i = 0 ; i < max_iter ; i++){

        local_sum = cblas_ddot(n,  x + offset, 1, x + offset, 1) ;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm) ;
        inv_norm = 1.0/sqrt(global_sum); // Calculate norm2
        cblas_dscal(global_n, inv_norm, x, 1);  // Normalise

        matrix_vector_csr(csr, x, x_local) ;
        MPI_Allgatherv(x_local, n, MPI_DOUBLE, x_new, recvcounts, displs, MPI_DOUBLE, comm);
        
        lambda_local = cblas_ddot(n, x_local, 1, x + offset, 1) ;
        MPI_Allreduce(&lambda_local, &lambda, 1, MPI_DOUBLE, MPI_SUM, comm) ;

        _ = x;
        x = x_new;
        x_new = _;
        
        if (rank == 0) {
            // lambda = cblas_ddot(global_n, x, 1, x_new, 1) ; // Calculate dot product
            // printf("%f\n", lambda);
        }
    }

    free(x_local) ;
    free(x_new) ;
    free(recvcounts) ;
    free(displs) ;
    
    return lambda;
}








