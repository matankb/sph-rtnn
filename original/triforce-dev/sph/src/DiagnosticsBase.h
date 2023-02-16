#ifndef TRIFORCE_DIAGNOSTICS_BASE_H
#define TRIFORCE_DIAGNOSTICS_BASE_H
/*
 * DiagnosticsBase.h
 *
 *
 *
 * Created On: 07/26/2022
 *
 * Last Updated:
 *    * MJL - 07/26/2022 - Initially created
 */

#include "Utilities.h"

class DiagnosticsBase {
public: // Constructors & Deconstructor
  DiagnosticsBase(const uint32_t size_input, const ParticleManagerBase* pm_input) : maxSize(size_input), pm(pm_input) {}
  virtual ~DiagnosticsBase() = default;

public:
  uint32_t maxSize;
  uint32_t size = 0;

protected: // Protected Class Data
  const ParticleManagerBase* pm;
};


#endif // TRIFORCE_DIAGNOSTICS_BASE_H
