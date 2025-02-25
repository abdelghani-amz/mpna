#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <string.h>


double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    matrix[0] = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 1; i < rows; i++) {
        matrix[i] = matrix[0] + i * cols;
    }
    return matrix;
}

void gmres(double** A, double* b, double* x, int nrows, int max_iter, double tol) {
    
    int m = max_iter;
    int n = nrows;
    double** V = allocate_matrix(m+1, nrows);
    memset(V[0], 0, (m+1) * nrows * sizeof(double));
    double** H = allocate_matrix(m+1, m);
    memset(H[0], 0, (m+1) * m * sizeof(double));

    double* Ax = (double*)malloc(n * sizeof(double));
    double* r = (double*)malloc(n * sizeof(double));
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0, A[0], m, x, 1, 0.0, Ax, 1);
    memcpy(r, b, n*sizeof(double));
    cblas_daxpy(n, -1.0, Ax, 1, r, 1);

    double beta = cblas_dnrm2(n, r, 1);
    cblas_dscal(n, 1.0/beta, r, 1);
    memcpy(V[0], r, n*sizeof(double));

    double* g = (double*)malloc((m+1) * sizeof(double));
    memset(g, 0, (m+1) * sizeof(double));
    g[0] = beta;

    double* cs = (double*)malloc(m * sizeof(double));
    double* sn = (double*)malloc(m * sizeof(double));
    int k;
    for (k = 0; k < m; k++) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, A[0], n, V[k], 1, 0.0, V[k+1], 1);
        for (int j = 0; j <= k; j++) {
            double norm = cblas_ddot(n, V[k+1], 1, V[j], 1);
            H[j][k] = norm;
            printf("norm = %f \n", norm);
            cblas_daxpy(n, -norm, V[j], 1, V[k+1], 1);
        }
        
        H[k+1][k] = cblas_dnrm2(n, V[k+1], 1);

        printf("H[k+1][k] %f \n", H[k+1][k]);
        cblas_dscal(n, 1.0/H[k+1][k], V[k+1], 1);
        
        for (int j = 0; j < nrows; j++) {
            printf("V[k+1] %f \n", V[k+1][j]);
        }
        
        
        for (int i = 0; i < k; i++) {
            double temp = cs[i] * H[i][k] + sn[i] * H[i+1][k];
            H[i+1][k] = -sn[i] * H[i][k] + cs[i] * H[i+1][k];
            H[i][k] = temp;
        }
        
        // Compute the new Givens rotation to eliminate H[k+1][k]
        double delta = sqrt(H[k][k]*H[k][k] + H[k+1][k]*H[k+1][k]);
        if (delta == 0.0) {
            cs[k] = 1.0;
            sn[k] = 0.0;
        } else {
            cs[k] = H[k][k] / delta;
            sn[k] = H[k+1][k] / delta;
        }
        
        // Apply the rotation to the current column of H
        H[k][k] = cs[k]*H[k][k] + sn[k]*H[k+1][k];
        H[k+1][k] = 0.0;
        
        // Update the g vector with the new Givens rotation
        double temp = cs[k]*g[k] + sn[k]*g[k+1];
        g[k+1] = -sn[k]*g[k] + cs[k]*g[k+1];
        g[k] = temp;
        
        // Check for convergence: if the residual (|g[k+1]|) is small enough, stop
        if (fabs(g[k+1]) < tol) {
            k++; // Adjust k to reflect the number of iterations performed
            break;
        }

    }

    double* y_ls = (double*)malloc(k * sizeof(double));
    for (int i = k - 1; i >= 0; i--) {
        y_ls[i] = g[i];
        for (int j = i + 1; j < k; j++) {
            y_ls[i] -= H[i][j] * y_ls[j];
        }
        y_ls[i] /= H[i][i];
    }
    
    // Update the solution: x = x + V * y_ls
    for (int i = 0; i < k; i++) {
        cblas_daxpy(n, y_ls[i], V[i], 1, x, 1);
    }
    for(int i = 0; i < n; i++) {
        printf("%f ", x[i]);
    }

    printf("\n");
    cblas_dgemv(CblasRowMajor , CblasNoTrans , n , m , 1.0 , A[0] , m , x , 1 , 0.0 , Ax , 1);
    for(int i = 0; i < n; i++) {
        printf("%f ", Ax[i]);
    }
    printf("\n");
}

int main(int argc, char const *argv[])
{
    int nrows = 3;
    int ncols = 3;
    double** A = (double**)malloc(nrows*sizeof(double*));
    A[0] = (double*)malloc(nrows * ncols * sizeof(double));
    for (int i = 1; i < nrows; i++) {
        A[i] = A[0] + i * ncols;
    }

    double* x = (double*)malloc(nrows*sizeof(double));
    memset(x, 0, nrows*sizeof(double));

    A[0][0] = 4.0; A[0][1] = 1.0; A[0][2] = 0.0;
    A[1][0] = 1.0; A[1][1] = 3.0; A[1][2] = 1.0;
    A[2][0] = 0.0; A[2][1] = 1.0; A[2][2] = 2.0;

    double b[] = {1.0, 2.0, 3.0};

    gmres(A, b, x, nrows, 10, 1e-6);
    printf("\n");
    return 0;
}
