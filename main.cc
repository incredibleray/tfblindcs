#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <iostream>
#include <armadillo>
#include "load.hpp"
#include <numeric>

arma::vec linreg(const arma::mat& A,
const arma::rowvec& y) {
    auto linreg=mlpack::regression::LinearRegression(A, y, 0, false);


auto beta=linreg.Parameters();

std::cout<< beta << std::endl;

    return beta;
}

mlpack::regression::LARS lasso(const arma::mat& A,
const arma::rowvec& y,
double lambda) {
    auto lar=mlpack::regression::LARS(false, lambda);

lar.Train(A, y);

// std::cout << lar.Beta();
auto beta=lar.Beta();
auto nonzeroBeta=arma::nonzeros(beta);

std::cout<< lambda << ": " << arma::size(nonzeroBeta) << std::endl;

    return lar;
}

int main() {

// prepareData();

arma::mat A;
arma::rowvec y;

load(A, y);
std::cout << A.n_rows<<std::endl<< A.n_cols<<std::endl << y.n_cols<<std::endl;


auto beta=linreg(A, y);

mlpack::data::Save("beta.csv", beta, true, false);

// uint N=200;
// arma::mat betaM=arma::zeros(N,A.n_rows+1);
// betaM.col(0)=arma::logspace(-4, 0, N);

// betaM.each_row( [&A, &y](arma::rowvec& a){ a.cols(1, a.n_cols-1)=lasso(A, y, a(0)).t(); } );
auto lar =lasso(A, y, 0.0001);
auto larPath=lar.BetaPath();

std::cout <<larPath.size()<<std::endl;
auto pathM=arma::mat();

for (const auto p : larPath) {
    pathM= arma::join_cols(pathM, p.t());
}

mlpack::data::Save("lasso_path.csv", pathM, true, false);


}