#ifndef TRIFORCE_IDEAL_GAS_EOS_H
#define TRIFORCE_IDEAL_GAS_EOS_H
/*
 * IdealGasEos.h
 *
 *
 *
 * Created On: 07/25/2022
 *
 * Last Updated:
 *    * MJL - 07/25/2022 - Initially created
 */

#include "Utilities.h"
#include "EosBase.h"

class IdealGasEos : public EosBase {
public: // Public Class Data
  fptype gamma;

public: // Constructors & Deconstructor
  explicit IdealGasEos( const fptype gamma_input ) : gamma (gamma_input) {};
  ~IdealGasEos() override = default;

public: // Class Functions
  KOKKOS_INLINE_FUNCTION fptype pFromRhoAndU( const fptype rho, const fptype u ) override {
    return (gamma - 1.0) * rho * u;
  }

  KOKKOS_INLINE_FUNCTION fptype soundSpeed( const fptype u ) const {
    return sqrt((gamma - 1.0) * fmax(0.0, u));
  }
};


#endif // TRIFORCE_IDEAL_GAS_EOS_H
