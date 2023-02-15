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
#include "optix.h"
#include "Parameters.h"
#include "constants.h"

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

  void updateNeighborList(int frame);
  void sortNeighborList(int frame);


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

  printf("max_neighbors = %d\n", params->sim.max_neighbors);
  printf("maxSize = %d\n", pm->maxSize);

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


// return true if a < b
bool compareNeighbors(neighbors* a, neighbors* b) {
  if (a->pid1 < b->pid1) {
    return true;
  } else if (a->pid1 > b->pid1) {
    return false;
  } else { // a.pid1 == b.pid1
    if (a->pid2 < b->pid2) {
      return true;
    }
    return false;
  }
}

void NeighborsBase::sortNeighborList(int frame) {
  return; // TEMPORARY
  
  ParticleManagerBase pmCopy = *pm;
  Kernel kernelCopy = *kernel;

  Kokkos::View<neighbors*>::HostMirror neighborList_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, neighborList);
  Kokkos::View<uint32_t>::HostMirror pairCounter_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pairCounter);

  // first normalize
  printf("Pair counter is: %d", pairCounter_h());
  printf("normalizing\n");
  for (int i = 0; i < pairCounter_h(); i++) {
    neighbors pair = neighborList_h(i);
    if (pair.pid1 < pair.pid2) {
      uint32_t pid2_temp = pair.pid2;
      pair.pid2 = pair.pid1;
      pair.pid1 = pid2_temp;

      // recompute dwdx, might be possible to just flip it
      Particle pi = pmCopy.getParticle(pair.pid1);
      Particle pj = pmCopy.getParticle(pair.pid2);
      fptype mhsml = (pi.hsml + pj.hsml) / 2.0;
      fptype w;
      vec3<fptype> dwdx;
      vec3<fptype> dx = pi.loc - pj.loc;
      fptype radius = dx.length();
      kernelCopy.computeKernel(w, dwdx, dx, radius, 0.5*(pi.hsml + pj.hsml));

      pair.w = w;
      pair.dwdx = dwdx;

      neighborList_h(i) = pair;
    }
  }
  printf("normalizing finished\n");

  // then sort
  for (int i = 0; i < pairCounter_h(); i++) {
    for (int j = i + 1; j < pairCounter_h(); j++) {
      if (compareNeighbors(&neighborList_h(j), &neighborList_h(i))) {
        neighbors temp = neighborList_h(i);
        neighborList_h(i) = neighborList_h(j);
        neighborList_h(j) = temp;
      }
    }
  }

  // copy back
  Kokkos::deep_copy(neighborList, neighborList_h);
}


void NeighborsBase::updateNeighborList(int frame) {
  float radius_limit = 0.040404;

  // reset to zero (don't need to reset neighborList)
  Kokkos::deep_copy(pairCounter, 0);
  Kokkos::deep_copy(interactionCount, 0);

  ParticleManagerBase pmCopy = *pm;
  Kernel kernelCopy = *kernel;

  auto startSearch = high_resolution_clock::now();

  if (ENABLE_RTNN) {
    fptype* points = new fptype[3 * pm->pNum];
    fptype* radii = new fptype[pm->pNum];
  
    // /*

    Kokkos::View<neighbors*>::HostMirror neighborList_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, neighborList);
    Kokkos::View<uint32_t*>::HostMirror interactionCount_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, interactionCount); // TODO: this copy might not be needed since it's been reset

    // copy points into a normal float
    // and calculate radii based on hsml and scale
    for (int i = 0; i < pm->pNum; i++) {
      auto pi = pmCopy.getParticle(i);
      
      fptype x = pi.loc.x();
      fptype y =  pi.loc.y();
      fptype z =  pi.loc.z();

      if (frame == 7 && i == 3) {
        // printf("Inside, p3_x  = %0.70f\n", i, x);
      }

      points[(i * 3)] = x;
      points[(i * 3) + 1] = y;
      points[(i * 3) + 2] = z;

      radii[i] = kernelCopy.scale_k * pi.hsml;
    }

    // manually fill 
    // points[0] = 0;
    // points[1] = 0;
    // points[2] = 0;
    // points[3] = 10;
    // points[4] = 10;
    // points[5] = 10;
    // points[6] = 0.00001;
    // points[7] = 0;
    // points[8] = 0;

    // printf("NUMBER OF POINTS %d", pm->pNum);
    
    // hand off to knn to get neighbor list
    int neighbors_count = pm->pNum;
    int max_interactions = pm->maxSize * params->sim.max_neighbors;
    fptype* a = new fptype[1];
    float** computed_neighbors = getNeighborList(points, radii, neighbors_count, radius_limit, max_interactions, frame);//)pm->pNum);

    // printf("\n======= NOW PRINTING NEIGHBORS LIST ====== \n\n");

    // int i = 40;
    // printf("%.20f , %.20f, %d\n\n", neighbors[0][0], points[21], neighbors[0][0] == points[21]);
    
    // not using a kokkos atomic for this because it gets converted back sequentially
    int pCount = 0;

    for (uint32_t i = 0; i < (neighbors_count * neighbors_count); i++) {
      float* current_neighbor = computed_neighbors[i];
      if (current_neighbor == NULL) { // computer_neighbors is null-terminated
        // printf("Neighbor is null!");
        // continue;
        break;
      }
      
      // get index of points
      uint32_t first_index = -1;
      uint32_t second_index = -1;
      for (uint32_t j = 0; j < neighbors_count; j++) {
        float x = points[(j * 3)];
        float y = points[(j * 3) + 1];
        float z = points[(j * 3) + 2];
        if (i == 0) {
          // printf("%0.5f, %0.5f, %0.5f\n", x, y, z);
        }
        if (current_neighbor[0] == x && current_neighbor[1] == y && current_neighbor[2] == z) {
          first_index = j;
        }
        if (current_neighbor[3] == x && current_neighbor[4] == y && current_neighbor[5] == z) {
          second_index = j;
        }
      }
      
      if (first_index == second_index) {
        continue;
      }

      int found_duplicate = 0;
      // deduplicate
      for (uint32_t k = 0; k < pCount; k++) {
        neighbors pair = neighborList_h(k);
        if (pair.pid1 == second_index && pair.pid2 == first_index) {
          // printf("Found duplicate!!\n");
          found_duplicate = 1;
          break;
        }
      }

      if (found_duplicate) {
        continue;
      }

      // printf("<%f, %f, %f>, <%f, %f, %f>\n", current_neighbor[0], current_neighbor[1], current_neighbor[2], current_neighbor[3], current_neighbor[4], current_neighbor[5]);
      // printf("INDICES: <%d, %d>\n", first_index, second_index);

      auto pi = pmCopy.getParticle(first_index);
      auto pj = pmCopy.getParticle(second_index);
      // printf("SOME DATA: %f \n", pi.loc.x());

      fptype mhsml = (pi.hsml + pj.hsml) / 2.0;
      fptype w;
      vec3<fptype> dwdx;
      vec3<fptype> dx = pi.loc - pj.loc;
      fptype radius = dx.length();
      kernelCopy.computeKernel(w, dwdx, dx, radius, 0.5*(pi.hsml + pj.hsml));

      if (frame == 2 && first_index == 1 && second_index == 2) {
        // printf("i = %d | %0.100f, %0.100f, %0.100f", first_index, pi.loc.x(), pi.loc.y(), pi.loc.z());
        // printf("j = %d | %0.100f, %0.100f, %0.100f", second_index, pj.loc.x(), pj.loc.y(), pj.loc.z());
        printf("i = %d | %0.100f\n", first_index, pi.hsml);
        printf("j = %d | %0.100f\n", second_index, pj.hsml);
        printf("Here we are!\n");
        printf("Computed W %0.100f\n", w);
        // printf()
      }
      // int newPCount = Kokkos::atomic_fetch_add(&pairCounter(), 1);

      neighborList_h(pCount) = {first_index, second_index, w, dwdx};
      // Kokkos::atomic_increment(&interactionCount(i));
      // interactionCount_h(first_index)
      pCount++;

      // free the memory
      free(computed_neighbors[i]);
      // printf("now we are here, right?");
    }

    // TODO: maybe have caching? but see my question below. this just resets neighbors list every time
    // neighborList(1) = 
    // TODO: max interactions

    free(computed_neighbors);

    printf("Setting the pair counter to: %d", pCount);
    // copy back to device views
    Kokkos::deep_copy(neighborList, neighborList_h);
    Kokkos::deep_copy(interactionCount, interactionCount_h);
    Kokkos::deep_copy(pairCounter, pCount);

    auto endSearch = high_resolution_clock::now();
    double searchTime = (duration_cast<microseconds>(endSearch - startSearch)).count();
    printf("Search time: %f", searchTime);

    return;
  }

  // nParticles()

  Kokkos::parallel_for(pmCopy.allParticlesPolicy(),
    KOKKOS_CLASS_LAMBDA(const uint32_t &i) {
      // printf("we are at frame = %d\n", frame);
      // printf("Max interactions: %d\n", maxInteractions());
      auto pi = pmCopy.getParticleDevice(i);

      // printf("SEARCHING THROUGH POINTS:\n");
      for (uint32_t j = 0; j < nParticles(); j++) {

        if (i != j) {
        if (pairCounter() >= maxInteractions()) { break; }
        auto pj = pmCopy.getParticleDevice(j);

        // printf("<%f, %f, %f>, <%f, %f, %f>\n", pi.loc.x(), pi.loc.y(), pi.loc.z(), pj.loc.x(), pj.loc.y(), pj.loc.z());

        vec3<fptype> dx = pi.loc - pj.loc;
        fptype radius = dx.length();
        if (frame == 3 && ((i == 38 && j == 42))) {
          // printf("Inside (38, 42)\n");
          // printf("radius = %.100f, limit = %.100f, within limit = %s\n", radius, radius_limit, radius <= (kernelCopy.scale_k_d() * pi.hsml) ? "true" : "false");
          // printf("pi = <%0.100f, %f, %f>, pj = <%0.100f, %f, %f>, dx length = %0.100f\n", pi.loc.x(), pi.loc.y(), pi.loc.z(), pj.loc.x(), pj.loc.y(), pj.loc.z(), radius);
        }
        // printf("Computed radius max: %f\n", kernelCopy.scale_k_d() * pi.hsml);

      // printf(pairCounter)

        if (radius <= radius_limit) {//(kernelCopy.scale_k_d() * pi.hsml)) {
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
            if (frame == 2 && i == 2 && j == 1) {
              printf("Here we are! Non-rtnn edition\n");
              // printf("i = %d | %0.100f, %0.100f, %0.100f", i, pi.loc.x(), pi.loc.y(), pi.loc.z());
              printf("i = %d | %0.100f", i, pi.hsml);
              printf("j = %d | %0.100f", j, pj.hsml);
              // printf("j = %d | %0.100f, %0.100f, %0.100f", j, pj.loc.x(), pj.loc.y(), pj.loc.z());
            }
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

  auto endSearch = high_resolution_clock::now();
  double searchTime = (duration_cast<microseconds>(endSearch - startSearch)).count();
  printf("Search time: %f", searchTime);
} // end updateNeighborList
#endif // TRIFORCE_NEIGHBORS_BASE_H
