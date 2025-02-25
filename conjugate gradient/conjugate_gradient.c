#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <string.h>



void conjugateGradient(double** A , double* x, double* b, int n, int m, int max_iter){
    
    double *r = (double*)malloc(n*sizeof(double));
    double *p = (double*)malloc(n*sizeof(double));
    double *Ap = (double*)malloc(n*sizeof(double));
    double *Ax = (double*)malloc(n*sizeof(double));
    double alpha, beta, rnorm;

    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0, A[0], m, x, 1, 0.0, Ax, 1);
    memcpy(r, b, n*sizeof(double));
    cblas_daxpy(n, -1.0, Ax, 1, r, 1);
    memcpy(p, r, n*sizeof(double));

    for (int i = 0; i < max_iter; i++) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0, A[0], m, p, 1, 0.0, Ap, 1);
        rnorm = cblas_ddot(n, r, 1, r, 1) ;
        alpha = rnorm / cblas_ddot(n, p, 1, Ap, 1);
        cblas_daxpy(n, alpha, p, 1, x, 1);
        cblas_daxpy(n, -alpha, Ap, 1, r, 1); 
        beta = cblas_ddot(n, r, 1, r, 1) / rnorm ;
        cblas_dscal(n, beta, p, 1); 
        cblas_daxpy(n, 1.0, r, 1, p, 1);

        if (cblas_dnrm2(n, r, 1) < 1e-8) {
            break;
        }
    }

    for(int i = 0; i < n; i++) {
        printf("%f ", x[i]);
    }

    printf("\n");
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0, A[0], m, x, 1, 0.0, Ax, 1);
    for(int i = 0; i < n; i++) {
        printf("%f ", Ax[i]);
    }
    printf("\n");

    free(r);
    free(p);
    free(Ap);
    free(Ax);
}


int main(int argc, char const *argv[])
{
    /* code */
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

    conjugateGradient(A, x, b, nrows, ncols, 1000);

    free(A[0]);
    free(A);
    free(x);

    return 0;
}
