#ifndef TRIFORCE_NEIGHBORS_BASE_H
#define TRIFORCE_NEIGHBORS_BASE_H
/*
 * NeighborsBase.h
 *
 * Abstract Base Class for neighbor finding
 *
 *
 *
 * Created On: 07/19/2022
 *
 * Last Modified:
 *    * AJK - 07/19/2022 - Initially created
 *    * MJL - 07/22/2022 - Filling in class
 */

#include "Utilities.h"
#include "Kernel.h"

struct neighbors {
  uint32_t pid1 = 0;
  uint32_t pid2 = 0;
  fptype w = 0.0;
  vec3<fptype> dwdx;
};


class NeighborsBase {
public:
  // Constructors & Destructors
  NeighborsBase(const Parameters* params_input, const ParticleManagerBase* pm_input);
  virtual ~NeighborsBase() = default;

  void updateNeighborList();


public:
  Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space> nParticles;
  Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space> maxNeighbors;
  Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space> maxInteractions;
  Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space> pairCounter;
  Kokkos::View<neighbors*, Kokkos::DefaultExecutionSpace::memory_space> neighborList;
  Kokkos::View<uint32_t*, Kokkos::DefaultExecutionSpace::memory_space> interactionCount;

  Kernel *kernel;

protected: // Protected Class Data
  const Parameters* params;
  const ParticleManagerBase* pm;
};

//
// ----- Constructors & Destructor -----
//
NeighborsBase::NeighborsBase(const Parameters* params_input, const ParticleManagerBase* pm_input) :
  params(params_input), pm(pm_input)
{
  nParticles = Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space>("Triforce::NeighborsBase::nParticles");
  Kokkos::deep_copy(nParticles, pm->pNum);

  maxNeighbors = Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space>("Triforce::NeighborsBase::maxNeighbors");
  Kokkos::deep_copy(maxNeighbors, params->sim.max_neighbors);

  maxInteractions = Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space>("Triforce::NeighborsBase::maxInterations");
  Kokkos::deep_copy(maxInteractions, pm->maxSize * params->sim.max_neighbors);

  pairCounter = Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space>("Triforce::NeighborsBase::pairCounter");
  Kokkos::deep_copy(pairCounter, 0);

  neighborList = Kokkos::View<neighbors*, Kokkos::DefaultExecutionSpace::memory_space>("Triforce::NeighborsBase::neighborList", pm->maxSize * params->sim.max_neighbors);

  interactionCount = Kokkos::View<uint32_t*, Kokkos::DefaultExecutionSpace::memory_space>("Triforce::NeighborsBase::interactionCount", pm->maxSize);
  Kokkos::deep_copy(interactionCount, 0);

  kernel = new Kernel(params->sim.smoothing_kernel);
}

void NeighborsBase::updateNeighborList() {
  // reset to zero (don't need to reset neighborList)
  Kokkos::deep_copy(pairCounter, 0);
  Kokkos::deep_copy(interactionCount, 0);

  ParticleManagerBase pmCopy = *pm;
  Kernel kernelCopy = *kernel;
  Kokkos::parallel_for(pmCopy.allParticlesPolicy(),
    KOKKOS_CLASS_LAMBDA(const uint32_t &i) {
      auto pi = pmCopy.getParticleDevice(i);

      for (uint32_t j = 0; j < nParticles(); j++) {
        if (i != j) {
        if (pairCounter() >= maxInteractions()) { break; }
        auto pj = pmCopy.getParticleDevice(j);

        vec3<fptype> dx = pi.loc - pj.loc;
        fptype radius = dx.length();

        if (radius <= (kernelCopy.scale_k_d() * pi.hsml)) {

          bool repeatedPair = false;
	  for (uint32_t k = 0; k < pairCounter(); k++) {
            neighbors n = neighborList(k);
	    if ((n.pid1 == i && n.pid2 == j) || (n.pid1 == j && n.pid2 == i)) {
	      repeatedPair = true;
	    }
	  }
          
	  if (!repeatedPair) {
            fptype mhsml = (pi.hsml + pj.hsml) / 2.0;
            fptype w;
            vec3<fptype> dwdx;
            kernelCopy.computeKernel(w, dwdx, dx, radius, 0.5*(pi.hsml + pj.hsml));
            int newPCount = Kokkos::atomic_fetch_add(&pairCounter(), 1);
            neighborList(newPCount) = {i, j, w, dwdx};

	    //printf(" %d %d %d \n",newPCount,i,j);
            Kokkos::atomic_increment(&interactionCount(i));
            Kokkos::atomic_increment(&interactionCount(j));
	  } // end if !repeatedPai
        } // end if r<h
	} // i!=j
      } // end for j=i+1
    } // end kokkos_lambda
  ); // end parallel_for

} // end updateNeighborList
#endif // TRIFORCE_NEIGHBORS_BASE_H
