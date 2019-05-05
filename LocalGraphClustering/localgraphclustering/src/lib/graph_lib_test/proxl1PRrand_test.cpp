#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <ctime>

#include "../include/proxl1PRrand_c_interface.h"

using namespace std;

int main() {
    int64_t ai[5] = {0,3,6,7,11};
    int64_t aj[11] = {0,1,3,0,1,3,3,0,1,2,3};
    double a[11] = {1,1,1,1,1,1,1,1,1,1,1};
    double rho = 0.000001;
    double alpha = 0.15;
    int64_t v[2] = {1,2};
    int64_t num_seed = 2;
    double epsilon = 0.0000001;
    int64_t maxiter = 10;
    double* p = new double[4]();
    double* y = new double[4]();
    double* grad = new double[4]();
    int64_t n = 4;
    double* d = new double[n];
    double* ds = new double[n];
    double* dsinv = new double[n];
    for(int i = 0; i < n; i ++){
        d[i] = ai[i+1] - ai[i];
        ds[i] = sqrt(d[i]);
        dsinv[i] = 1/ds[i];
    }
    double p0[4] = {0,0,0,0};
    bool not_converged;
    clock_t begin = clock();
    not_converged = proxl1PRaccel_rand64(n,ai,aj,a,alpha,rho,v,2,d,ds,dsinv,epsilon,grad,p,p0,maxiter,0,100, true);
    // not_converged = proxl1PRrand64(n,ai,aj,a,alpha,rho,v,2,d,ds,dsinv,epsilon,grad,p,p0,maxiter,0,100, true);
    clock_t end = clock();
    cout << (not_converged ? "not" : "") << "converged" << endl;
    double sum = 0;
    cout << "p: [ ";
    for(int i = 0; i < 4; i ++) {
        cout << p[i] << ' ';
        sum += p[i];
    }
    cout << "] ";
    cout << "total probability: " << sum << endl;
    cout << "gradient: ";
    for(int i = 0; i < 4; i ++){
        cout << grad[i] << ' ';
    }
    cout << '\n';
    cout << "time used: " << double(end - begin) << "ms" << endl;
    return EXIT_SUCCESS;
}