#ifndef TRIFORCE_EOS_BASE_H
#define TRIFORCE_EOS_BASE_H
/*
 * EosBase.h
 *
 *
 *
 * Created On: 07/25/2022
 *
 * Last Updated:
 *    * MJL - 07/25/2022 - Initially created
 */

#include "Utilities.h"

class EosBase {
public: // Constructors & Deconstructor
  EosBase() = default;
  virtual ~EosBase() = default;

public:
  virtual KOKKOS_INLINE_FUNCTION fptype pFromRhoAndU( const fptype rho, const fptype u ) { return 0.0; };
};


#endif // TRIFORCE_EOS_BASE_H
