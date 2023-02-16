#ifndef TRIFORCE_DIAGNOSTICS_H
#define TRIFORCE_DIAGNOSTICS_H
/*
 * Diagnostics.h
 *
 *
 *
 * Created On: 07/26/2022
 *
 * Last Updated:
 *    * MJL - 07/26/2022 - Initially created
 */

#include "Utilities.h"
#include "DiagnosticsBase.h"

class Diagnostics : public DiagnosticsBase {
public: // Public Class Data
  Kokkos::View<fptype*, Kokkos::HostSpace> kineticEnergy;
  Kokkos::View<fptype*, Kokkos::HostSpace> internalEnergy;
  fptype E0; // initial energy

public: // Constructors & Deconstructor
  Diagnostics(uint32_t size_input, const ParticleManagerBase* pm_input);
  ~Diagnostics() override = default;

public:
  void updateDiagnostics() {
    fptype KE;
    fptype IE;
        
    Kokkos::parallel_reduce("Triforce::Diagnostics::Reduce",
      Kokkos::RangePolicy<Kokkos::OpenMP>(0, pm->pNum),
      KOKKOS_LAMBDA(const uint32_t& pid, fptype& EC1) {
        EC1 += pm->getParticle(pid).calculateKE();
      }, // end kokkos_lambda
      KE
    ); // end parallel_reduce

    Kokkos::parallel_reduce("Triforce::Diagnostics::Reduce2",
      Kokkos::RangePolicy<Kokkos::OpenMP>(0, pm->pNum),
      KOKKOS_LAMBDA(const uint32_t& pid, fptype& EC1) {
        EC1 += pm->getParticle(pid).calculateIE();
      }, // end kokkos_lambda
      IE
    ); // end parallel_reduce

    kineticEnergy(size) = KE;
    internalEnergy(size) = IE;
    size++;
  } // end updateDiagnostics
};

//
// ----- Constructors & Destructor -----
//
Diagnostics::Diagnostics(const uint32_t size_input, const ParticleManagerBase* pm_input) :
  DiagnosticsBase(size_input, pm_input),
  kineticEnergy(Kokkos::View<fptype*, Kokkos::HostSpace>("Triforce::Diagnostics::kineticEnergy", size_input)),
  internalEnergy(Kokkos::View<fptype*, Kokkos::HostSpace>("Triforce::Diagnostics::internalEnergy", size_input))
{
  updateDiagnostics();
  E0 = kineticEnergy(0) + internalEnergy(0);
}


#endif // TRIFORCE_DIAGNOSTICS_H
