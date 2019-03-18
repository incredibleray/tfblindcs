#include "load.hpp"
#include <iostream>
#include <assert.h>

bool load(arma::mat& A, arma::rowvec& y) {
    arma::mat A;
  // Use data::Load() which transposes the matrix.
  // mlpack::data::Load("data.csv", data, true);
  mlpack::data::Load("X.csv", A, true);
  mlpack::data::Load("y.csv", y, true);

  return true;
//   mlpack::data::Save("y.csv", y, true);
}

arma::mat removeHeaderAndDate(const arma::mat& data){
   return data.rows(1, data.n_rows-1).cols(2, data.n_cols-1);
}

void prepareData(){
        arma::mat atod, etoh, x;
  // Use data::Load() which transposes the matrix.
  mlpack::data::Load("delta_price_symbols_A_to_D.csv", atod, true, false);
    mlpack::data::Load("delta_price_symbols_E_to_H.csv", etoh, true, false);
    mlpack::data::Load("delta_price_symbol_X.csv", x, true, false);

  atod=removeHeaderAndDate(atod);
  etoh=removeHeaderAndDate(etoh);
x=removeHeaderAndDate(x);

arma::mat data=arma::join_rows(arma::join_rows(atod,etoh), x);

arma::mat centered, normalized;
 mlpack::math::Center(data.t(), centered);
//   normalize(b, c);
normalized = arma::normalise(centered.t());

// mlpack::data::Save("joined_data.csv", data, true, false);
mlpack::data::Save("data.csv", normalized, true, false);

arma::mat q, r;
assert(arma::qr_econ(q, r, normalized.cols(0, normalized.n_cols-2)));
arma::mat qData=arma::join_rows(q, normalized.col(normalized.n_cols-1));
mlpack::data::Save("Qdata.csv", qData, true, false);

}


// int main() {
//     prepareData();
//     // load();
// }