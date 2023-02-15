#ifndef TRIFORCE_PARTICLE_H
#define TRIFORCE_PARTICLE_H
/*
 * Particle.h
 *
 * Abstract Base Class for particle data types (may or may not be necessary)
 *
 *
 * 
 * Created On: 07/19/2022
 *
 * Last Modified:
 *    * AJK - 07/19/2022 - Initially created
 *    * AJK - 07/21/2022 - Changed to hold pointers for integration w/
 *    manager
 */

#include "Utilities.h"
struct Particle {
  fptype mass;
  fptype rho;
  fptype p;
  fptype u;
  fptype c;
  fptype hsml;

  vec3<fptype> loc;
  vec3<fptype> vel;

  KOKKOS_FUNCTION Particle() = default;
  KOKKOS_FUNCTION Particle(fptype mass, fptype rho,
                           fptype p, fptype u,
                           fptype c, fptype hsml,
                           vec3<fptype> loc, vec3<fptype> vel);
  KOKKOS_FUNCTION Particle(const volatile Particle& rhs);
  KOKKOS_FUNCTION ~Particle() = default;

  KOKKOS_FUNCTION Particle& operator=(const Particle &rhs) {
    this->mass = rhs.mass;
    this->rho = rhs.rho;
    this->p = rhs.p;
    this->u = rhs.u;
    this->c = rhs.c;
    this->hsml = rhs.hsml;

    this->loc = rhs.loc;
    this->vel = rhs.vel;

    return *this;
  }

  KOKKOS_FUNCTION Particle& operator=(const volatile Particle &rhs) {
    this->mass = rhs.mass;
    this->rho = rhs.rho;
    this->p = rhs.p;
    this->u = rhs.u;
    this->c = rhs.c;
    this->hsml = rhs.hsml;

    this->loc = rhs.loc;
    this->vel = rhs.vel;

    return *this;
  }

  KOKKOS_FUNCTION void operator=(const Particle& rhs) volatile {}

  KOKKOS_FUNCTION fptype calculateKE() const { return 0.5 * mass * vel.length_squared(); }
  KOKKOS_FUNCTION fptype calculateIE() const { return mass * u;}

  friend std::ostream &operator<<(std::ostream &out, const Particle &p);
};

KOKKOS_FUNCTION Particle::Particle(fptype const mass, fptype const rho,
                                   fptype const p, fptype const u,
                                   fptype const c, fptype const hsml,
                                   vec3<fptype> const loc, vec3<fptype> const vel) :
  mass(mass), rho(rho), p(p), u(u), c(c), hsml(hsml), loc(loc), vel(vel)
{}

std::ostream &operator<<(std::ostream &out, const Particle &p) {
  return out << (p.loc)[0] << ", " << (p.loc)[1] << ", " << (p.loc)[2] << ", "
             << (p.vel)[0] << ", " << (p.vel)[1] << ", " << (p.vel)[2] << ", "
             <<  p.mass    << ", " <<  p.rho     << ", " <<  p.p       << ", "
             <<  p.u       << ", " <<  p.c       << ", " <<  p.hsml;
}

KOKKOS_FUNCTION Particle::Particle(const volatile Particle& rhs) {
  this->mass = rhs.mass;
  this->rho = rhs.rho;
  this->p = rhs.p;
  this->u = rhs.u;
  this->c = rhs.c;
  this->hsml = rhs.hsml;

  this->loc = rhs.loc;
  this->vel = rhs.vel;
}

#endif // TRIFORCE_PARTICLE_H
