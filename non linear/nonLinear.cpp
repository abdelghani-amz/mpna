
#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"


static const double kBase   = 0.01;   
static const double alpha    = 0.5;   
static const double sigmaVal = 0.1;   
static const double T0       = 1.0;   
static const double Qamp     = 1.0;   
static const double xCut     = 0.2;  


static const int    N        = 50;   
static const double L        = 1.0;   


double conductionCoeff(double T)
{
    double temp = (T>1.0e-14)? T : 1.0e-14;
    return kBase* std::pow(temp, alpha);
}


double flameSource(double x)
{
    return (x <= xCut)? Qamp: 0.0;
}


void TimeStepSystem(HYPRE_IJMatrix &A,
                         HYPRE_IJVector &rhs,
                         HYPRE_IJVector &xVec,
                         const std::vector<double> &Told,
                         double dt)
{
    double dx = L/double(N);
    double invDx2 = 1.0/(dx*dx);

    int low = 0;
    int high= N;

    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, low, high, low, high, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixSetMaxOffProcElmts(A,3);
    HYPRE_IJMatrixInitialize(A);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, low, high, &rhs);
    HYPRE_IJVectorSetObjectType(rhs, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(rhs);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, low, high, &xVec);
    HYPRE_IJVectorSetObjectType(xVec, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(xVec);

    for(int i=0; i<=N; i++){
        double rowVals[3]={0,0,0};
        int    colIdx[3]={0,0,0};
        int    count=0;

        if(i==0){
            
            double kRight= 0.5*( conductionCoeff(Told[i]) + conductionCoeff(Told[i+1]) );
            double diagVal= (1.0/dt) + 2.0*kRight*invDx2;

            rowVals[count]  = diagVal; colIdx[count++]= i;
            rowVals[count]  = -2.0*kRight*invDx2; colIdx[count++]= i+1;

            HYPRE_IJMatrixSetValues(A,1,&count,&i,colIdx,rowVals);

           
            double condOld= -2.0*kRight*invDx2*( Told[i+1]-Told[i] );
            double radPart= sigmaVal*( std::pow(Told[i],4) - std::pow(T0,4) );
            double source= flameSource(0.0);
            double leftover= ( (Told[i]-Told[i])/dt ) + condOld + radPart - source;
            double bVal= -leftover;
            HYPRE_IJVectorSetValues(rhs,1,&i,&bVal);
        }
        else if(i==N){
            rowVals[0]= 1.0;
            colIdx[0]= i;
            int one=1;
            HYPRE_IJMatrixSetValues(A,1,&one,&i,colIdx,rowVals);

            double leftover= (Told[i]-T0);
            double bVal= - leftover;
            HYPRE_IJVectorSetValues(rhs,1,&i,&bVal);
        }
        else {
            double kLeft= 0.5*( conductionCoeff(Told[i-1]) + conductionCoeff(Told[i]) );
            double kRight=0.5*( conductionCoeff(Told[i])   + conductionCoeff(Told[i+1]) );
            double diagVal= (1.0/dt) + (kLeft+ kRight)*invDx2;

           
            colIdx[count]= i-1; rowVals[count++]= -kLeft*invDx2;
            colIdx[count]= i;   rowVals[count++]= diagVal;
            colIdx[count]= i+1; rowVals[count++]= -kRight*invDx2;

            HYPRE_IJMatrixSetValues(A,1,&count,&i,colIdx,rowVals);

            double condOld= -( kRight*(Told[i+1]-Told[i]) - kLeft*(Told[i]-Told[i-1]) )*invDx2;
            double radPart= sigmaVal*( std::pow(Told[i],4) - std::pow(T0,4) );
            double source= flameSource(i*dx);
            double leftover= condOld + radPart - source;
            double bVal= - leftover;
            HYPRE_IJVectorSetValues(rhs,1,&i,&bVal);
        }
    }

    HYPRE_IJMatrixAssemble(A);
    HYPRE_IJVectorAssemble(rhs);
    HYPRE_IJVectorAssemble(xVec);
}


void solveBoomer(HYPRE_IJMatrix A,
                 HYPRE_IJVector b,
                 HYPRE_IJVector x)
{
    HYPRE_ParCSRMatrix parA=nullptr;
    HYPRE_ParVector    pb=nullptr, px=nullptr;

    HYPRE_IJMatrixGetObject(A, (void**)&parA);
    HYPRE_IJVectorGetObject(b, (void**)&pb);
    HYPRE_IJVectorGetObject(x, (void**)&px);

    HYPRE_Solver theSolver=nullptr;
    HYPRE_BoomerAMGCreate(&theSolver);
    HYPRE_BoomerAMGSetTol(theSolver, 1e-10);
    HYPRE_BoomerAMGSetMaxIter(theSolver, 200);
    HYPRE_BoomerAMGSetup(theSolver, parA, pb, px);
    HYPRE_BoomerAMGSolve(theSolver, parA, pb, px);
    HYPRE_BoomerAMGDestroy(theSolver);
}

void runTimeStepping(std::vector<double> &temp,
                     double dt,
                     int maxSteps,
                     double tol)
{
    for(int step=0; step< maxSteps; step++){
        HYPRE_IJMatrix A;
        HYPRE_IJVector b,x;
        TimeStepSystem(A,b,x, temp, dt);

        solveBoomer(A,b,x);

        // retrieve new solution
        std::vector<double> Tnew(temp.size());
        for(int i=0; i<(int)temp.size(); i++){
            double val;
            HYPRE_IJVectorGetValues(x,1,&i,&val);
            Tnew[i]= val;
        }

        // measure difference
        double diff2=0.0;
        for(int i=0; i<(int)temp.size(); i++){
            double d= (Tnew[i]- temp[i]);
            diff2+= d*d;
        }
        double diff= std::sqrt(diff2);

        // cleanup
        HYPRE_IJMatrixDestroy(A);
        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);

        temp= Tnew;
        if(diff< tol){
            std::cout<<"(Time-step) Converged at step="<<step<<"\n";
            break;
        }
    }
}


void ComputeResidual(std::vector<double>& F, const std::vector<double>& T, double dx) {
    double invDx2 = 1.0/(dx*dx);
    for(int i=0; i<=N; i++) {
        if(i == 0) {
            // Neumann BC: u_{-1} = u_1 (ghost point)
            double kRight = 0.5*(conductionCoeff(T[i]) + conductionCoeff(T[i+1]));
            double conduction = -2.0 * kRight * (T[i+1] - T[i]) * invDx2;
            F[i] = conduction + sigmaVal*(std::pow(T[i], 4) - std::pow(T0, 4)) - flameSource(i*dx);
        } else if(i == N) {
            // Dirichlet BC: T[N] = T0
            F[i] = T[i] - T0;
        } else {
            double kLeft = 0.5*(conductionCoeff(T[i-1]) + conductionCoeff(T[i]));
            double kRight = 0.5*(conductionCoeff(T[i]) + conductionCoeff(T[i+1]));
            double conduction = -(kRight*(T[i+1] - T[i]) - kLeft*(T[i] - T[i-1])) * invDx2;
            F[i] = conduction + sigmaVal*(std::pow(T[i], 4) - std::pow(T0, 4)) - flameSource(i*dx);
        }
    }
}

void BuildJacobian(HYPRE_IJMatrix &A, HYPRE_IJVector &b, const std::vector<double>& T, double dx) {
    double invDx2 = 1.0/(dx*dx);
    int ilower = 0, iupper = N;

    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);

    for(int i=0; i<=N; i++) {
        double rowVals[3] = {0.0, 0.0, 0.0};
        int colInds[3] = {0, 0, 0};
        int nnz = 0;

        if(i == 0) {
            // Neumann BC: derivatives at i=0
            double kRight = 0.5*(conductionCoeff(T[i]) + conductionCoeff(T[i+1]));
            double dkRight_du0 = 0.5 * alpha * kBase * std::pow(T[i], alpha-1);
            double dkRight_du1 = 0.5 * alpha * kBase * std::pow(T[i+1], alpha-1);

            // Diagonal term: ∂F₀/∂u₀
            rowVals[nnz] = 2.0 * invDx2 * (dkRight_du0*(T[i+1] - T[i]) + 2.0*kRight*invDx2)
                            + 4.0 * sigmaVal * std::pow(T[i], 3);
            colInds[nnz++] = i;

            // Off-diagonal: ∂F₀/∂u₁
            rowVals[nnz] = -2.0 * invDx2 * (dkRight_du1*(T[i+1] - T[i]) - 2.0*kRight*invDx2);
            colInds[nnz++] = i+1;

            HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, colInds, rowVals);
        } else if(i == N) {
            // Dirichlet BC: J[N][N] = 1
            rowVals[0] = 1.0;
            colInds[0] = i;
            nnz = 1;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, colInds, rowVals);
        } else {
            // Interior points
            double kLeft = 0.5*(conductionCoeff(T[i-1]) + conductionCoeff(T[i]));
            double kRight = 0.5*(conductionCoeff(T[i]) + conductionCoeff(T[i+1]));

            // Derivatives of κ with respect to u_{i-1}, u_i, u_{i+1}
            double dkLeft_duim1 = 0.5 * alpha * kBase * std::pow(T[i-1], alpha-1);
            double dkLeft_dui   = 0.5 * alpha * kBase * std::pow(T[i], alpha-1);
            double dkRight_dui = 0.5 * alpha * kBase * std::pow(T[i], alpha-1);
            double dkRight_duip1 = 0.5 * alpha * kBase * std::pow(T[i+1], alpha-1);

            // J[i][i-1]
            rowVals[nnz] = -invDx2 * (dkLeft_duim1*(T[i] - T[i-1]) - kLeft);
            colInds[nnz++] = i-1;

            // J[i][i]
            rowVals[nnz] = invDx2 * (dkLeft_dui*(T[i] - T[i-1]) + kLeft 
                            + dkRight_dui*(T[i+1] - T[i]) + kRight)
                            + 4.0 * sigmaVal * std::pow(T[i], 3);
            colInds[nnz++] = i;

            // J[i][i+1]
            rowVals[nnz] = -invDx2 * (dkRight_duip1*(T[i+1] - T[i]) - kRight);
            colInds[nnz++] = i+1;

            HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, colInds, rowVals);
        }

        // Set residual (RHS = -F)
        double Fi;
        ComputeResidual(Fi, T, dx, i); // Assume a helper function for F[i]
        HYPRE_IJVectorSetValues(b, 1, &i, &(-Fi));
    }

    HYPRE_IJMatrixAssemble(A);
    HYPRE_IJVectorAssemble(b);
}

void runNewtonMethod(std::vector<double>& T, int maxIters, double tol) {
    double dx = L / N;
    std::vector<double> F(T.size());
    ComputeResidual(F, T, dx);

    for(int iter=0; iter<maxIters; iter++) {
        HYPRE_IJMatrix J;
        HYPRE_IJVector rhs, delta;
        BuildJacobian(J, rhs, T, dx);

        solveBoomer(J, rhs, delta);

        // Extract delta and update T
        std::vector<double> deltaT(T.size());
        for(int i=0; i<=N; i++) {
            double val;
            HYPRE_IJVectorGetValues(delta, 1, &i, &val);
            deltaT[i] = val;
            T[i] += deltaT[i];
        }

        // Check convergence
        double normDelta = 0.0, normF = 0.0;
        for(double d : deltaT) normDelta += d*d;
        for(double f : F) normF += f*f;
        normDelta = std::sqrt(normDelta);
        normF = std::sqrt(normF);

        if(MPI_Comm_rank(MPI_COMM_WORLD, 0) == 0) {
            std::cout << "Newton Iter " << iter 
                      << ": Norm Delta=" << normDelta 
                      << ", Norm F=" << normF << std::endl;
        }

        if(normDelta < tol || normF < tol) {
            if(MPI_Comm_rank(MPI_COMM_WORLD, 0) == 0)
                std::cout << "Newton converged after " << iter << " iterations." << std::endl;
            break;
        }

        HYPRE_IJMatrixDestroy(J);
        HYPRE_IJVectorDestroy(rhs);
        HYPRE_IJVectorDestroy(delta);
    }
}



int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    HYPRE_Init();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==0){
        std::cout<<"---  Flame Diffusion Code  ---\n";
        std::cout<<"N="<<N<<", kBase="<<kBase<<", alpha="<<alpha
                 <<", sigma="<<sigmaVal<<", T0="<<T0<<"\n";
    }

    // Create initial guess
    std::vector<double> Tinit(N+1, T0);
    
    if(rank==0) std::cout<<"\n--- Running Pseudo-Time approach ---\n";
    std::vector<double> Ttime= Tinit;
    double dt= 0.001;     // or formula from README
    int steps= 500;
    double tol= 1e-6;
    runTimeStepping(Ttime, dt, steps, tol);

    if(rank==0){
        for(int i=0; i<=N; i+= (N>10 ? N/5 :1)){
            double x= double(i)/double(N);
            std::cout<<" x="<<x<<", T="<<Ttime[i]<<"\n";
        }
    }

    if(rank == 0) std::cout << "\n--- Running Newton's Method ---\n";
    std::vector<double> Tnewton = Tinit; // Or use Ttime from time-stepping
    int maxIters = 50;
    double tol = 1e-8;
    runNewtonMethod(Tnewton, maxIters, tol);

    if(rank == 0) {
        for(int i=0; i<=N; i += (N>10 ? N/5 : 1)) {
            double x = double(i)/double(N);
            std::cout << "x=" << x << ", T=" << Tnewton[i] << "\n";
        }
    }
    
    HYPRE_Finalize();
    MPI_Finalize();
    return 0;
}