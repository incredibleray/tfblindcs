#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <iostream>
#include <armadillo>
#include "load.hpp"
#include <numeric>

double linreg(const arma::mat& A,
const arma::rowvec& y) {
    auto linreg=mlpack::regression::LinearRegression(A, y, 0, false);


    return linreg.ComputeError(A, y);
}

double lasso(const arma::mat& A,
const arma::rowvec& y,
double lambda) {
    auto lar=mlpack::regression::LARS(false, lambda);

lar.Train(A, y);

    arma::rowvec yHat;
    lar.Predict(A, yHat);
    return arma::norm(y-yHat, 2);
}

int main() {

// prepareData();

arma::mat A;
arma::rowvec y;

load(A, y);
std::cout << A.n_rows<<std::endl<< A.n_cols<<std::endl << y.n_cols<<std::endl;


auto linregLoss=linreg(A, y);

auto lassoLoss =lasso(A, y, 0.0001);

std::cout <<"linregLoss: " << linregLoss <<"lassoLoss: " << lassoLoss <<std::endl;


}