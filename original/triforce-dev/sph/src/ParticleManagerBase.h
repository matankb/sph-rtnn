#ifndef TRIFORCE_PARTICLEMANAGER_BASE_H
#define TRIFORCE_PARTICLEMANAGER_BASE_H
/*
 * ParticleManagerBase.h
 *
 * Abstract Base Class for particle managers
 *
 *
 *
 * Created On: 07/19/2022
 *
 * Last Modified:
 *    * AJK - 07/19/2022 - Initially created
 */

#include "Utilities.h"
#include "Particle.h"

class ParticleManagerBase {
public:
  // Constructors & Destructors
  explicit ParticleManagerBase(uint32_t size);
  virtual ~ParticleManagerBase() = default;

  // Getters & Setters
  KOKKOS_FUNCTION Particle getParticle(uint32_t idx) const;
  KOKKOS_FUNCTION void setParticle(uint32_t idx,
          fptype massval, fptype rhoval,
          fptype pval, fptype uval,
          fptype cval, fptype hsmlval,
          vec3<fptype> locval, vec3<fptype> velval);
  KOKKOS_FUNCTION void setParticle(uint32_t idx, Particle &rhs);

  KOKKOS_FUNCTION Particle getParticleAtomic(uint32_t idx) const;
  KOKKOS_FUNCTION void setParticleAtomic(uint32_t idx,
          fptype massval, fptype rhoval,
          fptype pval, fptype uval,
          fptype cval, fptype hsmlval,
          vec3<fptype> locval, vec3<fptype> velval);
  KOKKOS_FUNCTION void setParticleAtomic(uint32_t idx, Particle &rhs);

  KOKKOS_FUNCTION virtual Particle getParticleDevice(uint32_t idx) const;

  Kokkos::RangePolicy<> allParticlesPolicy() const;
  Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace> allParticlesHostPolicy() const;

  inline void updateDeviceMirror() {
      Kokkos::deep_copy(particles_d, particles);
  }

public: // Public Class Data
  uint32_t maxSize = 1;
  uint32_t pNum;

public: // Protected Class Data
  Kokkos::View<Particle*, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::RandomAccess>> particles;
  Kokkos::View<Particle*, Kokkos::DefaultExecutionSpace::memory_space> particles_d;
};

//
// ----- Constructors & Destructor -----
//
ParticleManagerBase::ParticleManagerBase(const uint32_t size) :
  maxSize(size),
  pNum(0),
  particles(Kokkos::View<Particle*, Kokkos::DefaultHostExecutionSpace::memory_space, Kokkos::MemoryTraits<Kokkos::RandomAccess>>("Triforce::ParticleManagerBase::particles", size)),
  particles_d(Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space{}, particles))
{}

//
// ----- Kokkos Execution Policies -----
//
Kokkos::RangePolicy<> ParticleManagerBase::allParticlesPolicy() const {
  return Kokkos::RangePolicy<>(0, pNum);
}

Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace> ParticleManagerBase::allParticlesHostPolicy() const {
  return Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, pNum);
}

//
// ----- Getters & Setters -----
//
KOKKOS_FUNCTION Particle ParticleManagerBase::getParticle(const uint32_t idx) const {
  return particles(idx);
}

KOKKOS_FUNCTION void ParticleManagerBase::setParticle(const uint32_t idx,
        const fptype massval, const fptype rhoval,
        const fptype pval, const fptype uval,
        const fptype cval, const fptype hsmlval,
        const vec3<fptype> locval, const vec3<fptype> velval)
{
  particles(idx) = Particle(massval, rhoval, pval, uval, cval, hsmlval, locval, velval);
}

KOKKOS_FUNCTION void ParticleManagerBase::setParticle(uint32_t idx, Particle &rhs) {
  particles(idx) = rhs;
}

KOKKOS_FUNCTION Particle ParticleManagerBase::getParticleAtomic(const uint32_t idx) const {
  return Kokkos::atomic_load(&particles(idx));
}

KOKKOS_FUNCTION void ParticleManagerBase::setParticleAtomic(const uint32_t idx,
        const fptype massval, const fptype rhoval,
        const fptype pval, const fptype uval,
        const fptype cval, const fptype hsmlval,
        const vec3<fptype> locval, const vec3<fptype> velval)
{
  Kokkos::atomic_store(&particles(idx), Particle(massval, rhoval, pval, uval, cval, hsmlval, locval, velval));
}

KOKKOS_FUNCTION void ParticleManagerBase::setParticleAtomic(uint32_t idx, Particle &rhs) {
  Kokkos::atomic_store(&particles(idx), rhs);
}

KOKKOS_FUNCTION Particle ParticleManagerBase::getParticleDevice(uint32_t idx) const {
  return particles_d(idx);
}

#endif // TRIFORCE_PARTICLEMANAGER_BASE_H
