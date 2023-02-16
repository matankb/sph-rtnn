#ifndef TRIFORCE_SPH_BASE_H
#define TRIFORCE_SPH_BASE_H
/*
 * SimulationBase.h
 *
 *
 *
 * Created On: 07/25/2022
 *
 * Last Updated:
 *    * MJL - 07/25/2022 - Initially created
 */

#include "Utilities.h"

#include "SimulationBase.h"
#include "ParticleManagerBase.h"
#include "Diagnostics.h"
#include "IdealGasEos.h"
#include "NeighborsBase.h"


class SPH : public SimulationBase {
  public:
    // Constructors & Deconstructor
    explicit SPH(Parameters* params_input);
    ~SPH() override;

    // Methods
    void initStep( fptype dt );
    void report_start( uint32_t step, fptype time ) override;
    void report_end( uint32_t step ) override;
    void step( fptype dt ) override;
    void zeroArrays();
    void sumDensity();
    void updateHsml();
    void artificialViscosity();
    void internalForces();
    void initializeParticles();
    void advanceParticlesHalfStep( fptype dt );
    void advanceParticles( fptype dt );

  public: // Public Class Data
    ParticleManagerBase* pm;
    Diagnostics* diagnostics;
    IdealGasEos* eos;
    //  ChainingMesh* cm;
    NeighborsBase* neighborFinder;

    Kokkos::View<fptype, Kokkos::DefaultHostExecutionSpace::memory_space> minDt; // min(h_i/c_i)
    Kokkos::View<uint32_t>::HostMirror pairCounter_h;
    Kokkos::View<neighbors*>::HostMirror neighborList_h;
    Kokkos::View<uint32_t*>::HostMirror interactionCount_h;

  protected:
    uint32_t maxSize;

    using fpView_h = Kokkos::View<fptype*, Kokkos::DefaultHostExecutionSpace::memory_space>;
    fpView_h uMin;
    fpView_h avdudt;
    fpView_h ahdudt;
    fpView_h dudrho;
    fpView_h drho;
    fpView_h du;
    fpView_h dedt;
    fpView_h fpBuffer; // floating point buffer of 2xMaxSize
    fpView_h divV;

    using fpVecView_h = Kokkos::View<vec3<fptype>*, Kokkos::DefaultHostExecutionSpace::memory_space>;
    fpVecView_h av;
    fpVecView_h vMin;
    fpVecView_h dv;
    fpVecView_h indvdt;
    fpVecView_h ardvdt;
    fpVecView_h dvdt;
    fpVecView_h gradP;
    fpVecView_h fpBufferV; // vec3 floating point buffer
};

SPH::SPH(Parameters* params_input) : SimulationBase(params_input) {
  vec3<uint32_t> np = params->p1.num_particles; // number of particles
  const uint32_t npttl = np[0] * np[1] * np[2];

  eos = new IdealGasEos(params->p1.gamma);
  pm = new ParticleManagerBase(2 * npttl);

  minDt = Kokkos::View<fptype, Kokkos::HostSpace>("SPH::minDt");
  minDt() = 1.0e8;

  initializeParticles();

  if (params->time.enableCFL) {
    params->time.dt = params->time.CFL * minDt();
  }

  diagnostics = new Diagnostics( params->time.Nt+1, pm );
  neighborFinder = new NeighborsBase( params, pm );
  //    cm = new ChainingMesh(params, pm);
  //    cm->fillInChainingMesh();


  maxSize = pm->maxSize; // same as npttl, repetitive?

  uMin      = fpView_h("Triforce::SPH::uMin", maxSize);
  avdudt    = fpView_h("Triforce::SPH::avdudt", maxSize);
  ahdudt    = fpView_h("Triforce::SPH::ahdudt", maxSize);
  dudrho    = fpView_h("Triforce::SPH::dudrho", maxSize);
  drho      = fpView_h("Triforce::SPH::drho", maxSize);
  du        = fpView_h("Triforce::SPH::du", maxSize);
  dedt      = fpView_h("Triforce::SPH::dedt", maxSize);
  divV      = fpView_h("Triforce::SPH::divV", maxSize);
  fpBuffer  = fpView_h("Triforce::SPH::fpBuffer", 2 * maxSize);

  av        = fpVecView_h("Triforce::SPH::av", maxSize);
  vMin      = fpVecView_h("Triforce::SPH::vMin", maxSize);
  dv        = fpVecView_h("Triforce::SPH::dv", maxSize);
  indvdt    = fpVecView_h("Triforce::SPH::indvdt", maxSize);
  ardvdt    = fpVecView_h("Triforce::SPH::ardvdt", maxSize);
  dvdt      = fpVecView_h("Triforce::SPH::dvdt", maxSize);
  gradP     = fpVecView_h("Triforce::SPH::gradP", maxSize);
  fpBufferV = fpVecView_h("Triforce::SPH::fpBufferV", 2 * maxSize);
}

SPH::~SPH() {
  delete neighborFinder;
  delete diagnostics;
  delete pm;
  delete eos;
}

void SPH::initializeParticles() {
  vec3<uint32_t> np = params->p1.num_particles; // number of particles
  vec3<fptype> loc_den;
  for (int i = 0; i < 3; i++) {
    // loc_den[i] = (np[i] - 1 == 0) ? static_cast<fptype>(np[i]) : static_cast<fptype>(np[i] - 1);
    loc_den[i] = (np[i] - 1 == 0) ? static_cast<fptype>(2) : static_cast<fptype>(np[i] - 1);
  }

  Domain domain = params->p1.init_region;

  const uint32_t npttl = np[0] * np[1] * np[2];
  const fptype dx = domain.x1 - domain.x0;
  const fptype dy = domain.y1 - domain.y0;
  const fptype dz = domain.z1 - domain.z0;
  const fptype particleVolume = DIM > 1 ? dx * dy * dz / (npttl-1) : dx / (np[0] - 1.0);

  Kokkos::parallel_reduce("SPH::initializeParticles::loop2",
    Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>({0,0,0}, {np[0],np[1],np[2]}),
    KOKKOS_LAMBDA(const uint32_t &i, const uint32_t &j, const uint32_t &k, fptype &dt) {

      Particle tmp;
      tmp.rho = params->p1.mass_density;
      tmp.u = params->p1.internal_energy;
      tmp.p = eos->pFromRhoAndU( tmp.rho, tmp.u );
      tmp.c = eos->soundSpeed( tmp.u );
      tmp.mass = tmp.rho * particleVolume;
      tmp.hsml = 2.0 * pow(particleVolume, 1.0 / fpDIM);

      const auto Lloc0 = domain.x0 + static_cast<fptype>(i) / loc_den[0] * dx;
      const auto Lloc1 = domain.y0 + static_cast<fptype>(j) / loc_den[1] * dy;
      const auto Lloc2 = domain.z0 + static_cast<fptype>(k) / loc_den[2] * dz;
      tmp.loc = vec3<fptype>(Lloc0, Lloc1, Lloc2);

      tmp.vel = params->p1.velocity;

      const uint32_t pid = k + np[2]*j + np[2]*np[1]*i;
      pm->setParticle(pid, tmp);

      dt = fmin(dt, 0.5 * tmp.hsml / tmp.c);
    }, 
    Kokkos::Min<fptype>(minDt()) // end kokkos_lambda
  ); // end parallel_reduce
  pm->pNum += npttl;
} // end initializeParticles


void SPH::initStep( const fptype dt ) {
  zeroArrays();

  pm->updateDeviceMirror();

  auto startSearch = high_resolution_clock::now();
  neighborFinder->updateNeighborList();
  auto endSearch = high_resolution_clock::now();
  double searchTime = (duration_cast<microseconds>(endSearch - startSearch)).count();
  printf("== Search time: %f ==\n", searchTime);

  pairCounter_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, neighborFinder->pairCounter);
  neighborList_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, neighborFinder->neighborList);
  interactionCount_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, neighborFinder->interactionCount);

  sumDensity();

  updateHsml();

  if (params->sim.artificial_viscosity) { artificialViscosity(); }

  Kokkos::deep_copy(ardvdt, dvdt);
  Kokkos::deep_copy(avdudt, dedt);

  internalForces();

  advanceParticles( dt );

  diagnostics->updateDiagnostics();
}

void SPH::step( const fptype dt ) {
  zeroArrays();

  advanceParticlesHalfStep( dt ); // skip in first step

  pm->updateDeviceMirror();

  auto startSearch = high_resolution_clock::now();
  neighborFinder->updateNeighborList();
  auto endSearch = high_resolution_clock::now();
  double searchTime = (duration_cast<microseconds>(endSearch - startSearch)).count();
  printf("== Search time: %f ==\n", searchTime);

  pairCounter_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, neighborFinder->pairCounter);
  neighborList_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, neighborFinder->neighborList);
  interactionCount_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, neighborFinder->interactionCount);

  sumDensity();
  updateHsml();

  if (params->sim.artificial_viscosity) { artificialViscosity(); }

  Kokkos::deep_copy(ardvdt, dvdt);
  Kokkos::deep_copy(avdudt, dedt);

  internalForces();

  advanceParticles( dt );

  diagnostics->updateDiagnostics();
}


void SPH::report_start(uint32_t step, fptype time) {
  std::cout << "---------------------------------------------------" << std::endl;
  std::cout << "Starting step: " << step << ", at time = " << time << std::endl;
}

void SPH::report_end(uint32_t step) {
  fptype KEtmp = diagnostics->kineticEnergy(diagnostics->size - 1);
  fptype IEtmp = diagnostics->internalEnergy(diagnostics->size - 1);
  fptype E0tmp = diagnostics->E0;
  std::cout << "KE = " << KEtmp << std::endl;
  std::cout << "IE = " << IEtmp << std::endl;
  std::cout << "E0 = " << E0tmp << std::endl;

  fptype E_div_E0 = (KEtmp + IEtmp) / E0tmp;
  std::cout << "Completed step: " << step << " (KE+IE)/E0 = " << E_div_E0 << std::endl;
  std::cout << "---------------------------------------------------" << std::endl;
}

void SPH::zeroArrays() {
  Kokkos::deep_copy(uMin, 0.0);
  Kokkos::deep_copy(avdudt, 0.0);
  Kokkos::deep_copy(ahdudt, 0.0);
  Kokkos::deep_copy(drho, 0.0);
  Kokkos::deep_copy(du, 0.0);
  Kokkos::deep_copy(fpBuffer, 0.0);

  Kokkos::deep_copy(av, vec3<fptype>());
  Kokkos::deep_copy(vMin, vec3<fptype>());
  Kokkos::deep_copy(dv, vec3<fptype>());
  Kokkos::deep_copy(indvdt, vec3<fptype>());
  Kokkos::deep_copy(ardvdt, vec3<fptype>());
  Kokkos::deep_copy(dvdt, vec3<fptype>());
  Kokkos::deep_copy(fpBufferV, vec3<fptype>());
}

void SPH::sumDensity() {
  constexpr bool normalizeDensity = false; // nor_density is not set in yorick
  const uint32_t npttl = pm->pNum;
  vec3<fptype> hv = {0.0, 0.0, 0.0};
  fptype r = 0.0;

  Kokkos::deep_copy(fpBuffer, 0.0);

  if constexpr (normalizeDensity) {
    Kokkos::parallel_for("Triforce::SPH::sumDensity::normLoop1",
      pm->allParticlesHostPolicy(),
      KOKKOS_LAMBDA(const uint32_t i) {
        Particle tmp = pm->getParticle(i);
        fptype w = 0.0;
        vec3<fptype> dwdx;
        neighborFinder->kernel->computeKernel(w, dwdx, hv, r, tmp.hsml);
        fpBuffer(i) = w * tmp.mass / (PTINY + tmp.rho);
      } // end kokkos_lambda
    ); // end parallel_for

    Kokkos::parallel_for("Triforce::SPH::sumDensity::normLoop2",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, pairCounter_h()),
      KOKKOS_LAMBDA(const uint32_t ipair) {
        const neighbors n = neighborList_h(ipair);
        Particle p1 = pm->getParticle(n.pid1);
        Particle p2 = pm->getParticle(n.pid2);

        Kokkos::atomic_add(&fpBuffer(n.pid1), p2.mass / (PTINY + p2.rho) * n.w);
        Kokkos::atomic_add(&fpBuffer(n.pid2), p1.mass / (PTINY + p1.rho) * n.w);
      } // end kokkos_lambda
    ); // end parallel_for
  } // endif(normalizeDensity)

  Kokkos::parallel_for( "SPH::sumDensity::loop1",
    pm->allParticlesHostPolicy(),
    KOKKOS_LAMBDA(const uint32_t &pid) {
      Particle tmp = pm->getParticle(pid);
      fptype w = 0.0;
      vec3<fptype> dwdx = {0.0, 0.0, 0.0};
      neighborFinder->kernel->computeKernel(w, dwdx, hv, r, tmp.hsml);
      tmp.rho = w*tmp.mass;
      pm->setParticle(pid, tmp);
    } // end kokkos_lambda
  ); // end parallel_for

  Kokkos::parallel_for("SPH::sumDensity::loop2",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, pairCounter_h()),
    KOKKOS_LAMBDA(const uint32_t &ipair) {
      const neighbors n = neighborList_h(ipair);
      Particle p1 = pm->getParticleAtomic(n.pid1);
      Particle p2 = pm->getParticleAtomic(n.pid2);

      Kokkos::atomic_add(&drho(n.pid1), n.w * p2.mass);
      Kokkos::atomic_add(&drho(n.pid2), n.w * p1.mass);

    } // end kokkos_lambda
  ); // end parallel_for
  
  Kokkos::parallel_for("SPH::sumDensity::loop3",
    pm->allParticlesHostPolicy(),
    KOKKOS_LAMBDA(const uint32_t &pid) {
      Particle tmp = pm->getParticle(pid);
      tmp.rho = tmp.rho + drho(pid);
      pm->setParticle(pid, tmp);
    } // end kokkos_lambda
  ); // end parallel_for

  if constexpr (normalizeDensity) {
    Kokkos::parallel_for("SPH::sumDensity::normLoop3",
      pm->allParticlesHostPolicy(),
      KOKKOS_LAMBDA(const uint32_t &pid) {
        Particle tmp = pm->getParticle(pid);
        tmp.rho = tmp.rho / fpBuffer[pid];
        pm->setParticle(pid, tmp);
      } // end kokkos_lambda
    ); // end parallel_for
  } // endif(normalizeDensity)
} // end sumDensity

void SPH::updateHsml() {
  switch (params->sim.smoothing_length_evolution) {
    case 0:
      break; // end case 0
    case 1:
      Kokkos::parallel_for("Triforce::SPH::updateHsml::switch1",
        pm->allParticlesHostPolicy(),
        KOKKOS_LAMBDA(const uint32_t &pid) {
          Particle tmp = pm->getParticle(pid);
          tmp.hsml = 2.0 * pow(tmp.mass / tmp.rho, 1.0 / fpDIM);
          pm->setParticle(pid, tmp);
        }
      );
      break; // end case 1
    case 2:
      Kokkos::deep_copy(fpBuffer, 0.0);
      Kokkos::parallel_for("Triforce::SPH::updateHsml::switch2l1",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, pairCounter_h()),
        KOKKOS_LAMBDA(const uint32_t ipair) {
          const neighbors n = neighborList_h(ipair);
          Particle p1 = pm->getParticle(n.pid1);
          Particle p2 = pm->getParticle(n.pid2);

          const vec3 dvL = p2.vel - p1.vel;
          const fptype hvcc = dot(dvL, n.dwdx);

          Kokkos::atomic_add(&fpBuffer(n.pid1), p2.mass * hvcc / (PTINY + p2.rho));
          Kokkos::atomic_add(&fpBuffer(n.pid2), p1.mass * hvcc / (PTINY + p1.rho));
        }
      );

      const fptype dt = params->time.dt;
      Kokkos::parallel_for("Triforce::SPH::updateHsml::switch2l2",
        pm->allParticlesHostPolicy(),
        KOKKOS_LAMBDA(const uint32_t pid) {
          Particle tmp = pm->getParticle(pid);
          const fptype dhsml = fpBuffer(pid) * tmp.hsml / fpDIM;
          const fptype hsmlNew = tmp.hsml + dt * dhsml;

          tmp.hsml = (hsmlNew <= 0.0) ? tmp.hsml : hsmlNew;
          pm->setParticle(pid, tmp);
        }
      );
      break; // end case 2
  } // end switch
} // end updateHsml

void SPH::artificialViscosity() {
  const fptype alpha = 1.0; // shear viscosity
  const fptype  beta = 1.0; // bulk viscosity
  const fptype  etq2 = 0.01; // parameter to avoid singularities (etq**2)

  Kokkos::parallel_for("Triforce::SPH::artificialViscosity::loop1",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, pairCounter_h()),
    KOKKOS_LAMBDA(const uint32_t &ipair) {
      const neighbors n = neighborList_h(ipair);
      Particle p1 = pm->getParticle(n.pid1);
      Particle p2 = pm->getParticle(n.pid2);

      vec3<fptype> dx, dvL;
      dvL = p1.vel - p2.vel;
      dx = p1.loc - p2.loc;
      const fptype vr = dot(dvL,dx);
      const fptype rr = dot(dx,dx);
      const fptype mhsml = (p1.hsml + p2.hsml) / 2.0;

      // Artificial viscous force only if v_ij*r_ij < 0
      if ( vr < 0.0 ) {
        const fptype muv = mhsml * vr / (rr + SQR<fptype>(mhsml) * etq2);
        const fptype mc = (p1.c + p1.c) / 2.0;
        const fptype mrho = (p1.rho + p2.rho) / 2.0;
        const fptype piv = (beta * muv - alpha * mc) * muv / (PTINY + mrho);

        // Calculate SPH sum for artifical viscous force
        const vec3<fptype> h = n.dwdx * fptype(-1.0) * piv;
        Kokkos::atomic_add(&dvdt(n.pid1), h * p2.mass);
        Kokkos::atomic_sub(&dvdt(n.pid2), h * p1.mass);

        Kokkos::atomic_sub(&dedt(n.pid1), p2.mass * dot(dvL, h)); // AJK - 08/23/2022 - Are these two lines the correct operations? They seem fishy to me...
        Kokkos::atomic_sub(&dedt(n.pid2), p1.mass * dot(dvL, h));
      } // endif(vr)
    } // end kokkos_lambda
  ); // end parallel_for

  Kokkos::parallel_for("Triforce::artificialViscosity::loop2",
    pm->allParticlesHostPolicy(),
    KOKKOS_LAMBDA(const uint32_t &pid) {
      dedt(pid) *= 0.5;
    }
  );
} // end artificialViscosity

void SPH::internalForces() {
  const uint32_t npttl = pm->pNum;

  Kokkos::deep_copy(du, 0.0);
  Kokkos::deep_copy(divV, 0.0);
  Kokkos::deep_copy(fpBuffer, 0.0);
  Kokkos::deep_copy(dedt, 0.0);

  Kokkos::deep_copy(gradP, vec3<fptype>());
  Kokkos::deep_copy(dvdt, vec3<fptype>());

  // update minDt along with csound
  minDt() = 1.0e8;

  Kokkos::parallel_reduce("Triforce::SPH::internalForces::loop1",
    pm->allParticlesHostPolicy(),
    KOKKOS_LAMBDA(const uint32_t &pid, fptype &min_dt) {
      Particle tmp = pm->getParticle(pid);

      tmp.p = eos->pFromRhoAndU( tmp.rho, tmp.u );
      tmp.c = eos->soundSpeed( tmp.u );
      pm->setParticle(pid, tmp);

      fptype dt_tmp = tmp.hsml / tmp.c;
      min_dt = dt_tmp < min_dt ? dt_tmp : min_dt;
    }, 
    Kokkos::Min<fptype>(minDt) // end kokkos_lambda
  ); // end parallel_reduce

  Kokkos::parallel_for("Triforce::SPH::internalForces::loop2",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, pairCounter_h()),
    KOKKOS_LAMBDA(const uint32_t &ipair) {
      const neighbors n = neighborList_h(ipair);
      const uint32_t i = n.pid1;
      const uint32_t j = n.pid2;
      
      const Particle pi = pm->getParticleAtomic(i);
      const Particle pj = pm->getParticleAtomic(j);

      const vec3<fptype> local_dv = pi.vel - pj.vel;

      const vec3<fptype> h = n.dwdx * fptype(-1.0) * ( pi.p / (PTINY + SQR(pi.rho))
                             + pj.p / (PTINY + SQR(pj.rho)) );
      const fptype he = dot( -local_dv, h );

      if ( AFLAG == 1) {
        const vec3<fptype> dx = pi.loc - pj.loc;
        const fptype dp = pi.p - pj.p;
        const fptype drho = pi.rho - pj.rho;
        const fptype local_du = pi.u - pj.u;

        // minus sign cancels in yorick script: (-dv) / (-dx) = dv / dx

        for (uint32_t d = 0; d < DIM; d++) {
          if ((DIM == 1 && dx[d] != 0.0) || (DIM > 1 && dx[d] > 1e-4)) { // temporary fix for 2D and 3D
            Kokkos::atomic_add(&divV(i), local_dv[d] / dx[d] * n.w);
            Kokkos::atomic_add(&divV(j), local_dv[d] / dx[d] * n.w);

            Kokkos::atomic_add(&gradP(i)[d], dp / dx[d] * n.w);
            Kokkos::atomic_add(&gradP(j)[d], dp / dx[d] * n.w);
          } //else {
          //gradP(i)[d] += 0.0;
          //gradP(j)[d] += 0.0;
          //}
        }

        if (drho != 0.0) {
          Kokkos::atomic_add(&dudrho(i), local_du / drho * n.w);
          Kokkos::atomic_add(&dudrho(j), local_du / drho * n.w);
        }

        Kokkos::atomic_add(&fpBuffer(i), n.w);
        Kokkos::atomic_add(&fpBuffer(j), n.w);
      } //endif(AFLAG)

      Kokkos::atomic_add(&dvdt(i), h * pj.mass);
      Kokkos::atomic_sub(&dvdt(j), h * pi.mass);

      Kokkos::atomic_add(&dedt(i), he * pj.mass); // AJK - 08/23/2022 - another fishy spot
      Kokkos::atomic_add(&dedt(j), he * pi.mass);
    } // end kokkos_lambda
  ); // end parallel_for

  Kokkos::parallel_for("Triforce::SPH::internalForces::loop3",
    pm->allParticlesHostPolicy(),
    KOKKOS_LAMBDA(const uint32_t &pid) {
      dedt(pid) *= 0.5;
    } // end kokkos_lambda
  ); // end parallel_for

  if (AFLAG == 0) {
    Kokkos::parallel_for("Triforce::SPH::internalForces::loop4",
      pm->allParticlesHostPolicy(),
      KOKKOS_LAMBDA(const uint32_t &pid) {
        dv(pid) = dvdt(pid) + ardvdt(pid);
        du(pid) = dedt(pid) + avdudt(pid);
      } // end kokkos_lambda
    ); // end parallel_for
  } else if (AFLAG == 1) {
    Kokkos::parallel_for("Triforce::SPH::internalForces::loop5:",
      pm->allParticlesHostPolicy(),
      KOKKOS_LAMBDA(const uint32_t &pid) {
        gradP(pid) = gradP(pid) / (PTINY + fpBuffer(pid));
        dudrho(pid) = dudrho(pid) / (PTINY + fpBuffer(pid));
        divV(pid) = divV(pid) / (PTINY + fpBuffer(pid));
      } // end kokkos_lambda
    ); // end parallel_for
  } // endif(AFLAG)
} // end internalForces

void SPH::advanceParticlesHalfStep( const fptype dt ) {
  const fptype dtH = 0.5 * dt;

  // Advance the internal energies, velocities
  if (AFLAG == 0) {
  Kokkos::parallel_for("Triforce::SPH::advanceParticlesHalfStep::loop1",
    pm->allParticlesHostPolicy(),
    KOKKOS_LAMBDA(const uint32_t &pid) {
      Particle tmp = pm->getParticle(pid);
      uMin(pid) = tmp.u;

      fptype uNew;
      uNew = max(tmp.u + dtH * du(pid), fptype(0.0));

      vec3<fptype> vNew; //, xNew;
      vMin(pid) = tmp.vel;
      vNew = tmp.vel + dtH * dv(pid);

      tmp.u = uNew;
      tmp.vel = vNew;
      pm->setParticle(pid, tmp);
    } // end kokkos_lambda
  ); // end parallel_for

  } else {
    Kokkos::parallel_for("Triforce::SPH::advanceParticlesHalfStep::loop2",
      pm->allParticlesHostPolicy(),
      KOKKOS_LAMBDA(const uint32_t &pid) {
        Particle tmp = pm->getParticle(pid);

        fptype uNew;
        uNew = tmp.u + dtH * (fptype(-1.0) * tmp.p / tmp.rho * divV(pid));  // divV is a vector!
        // + CU1 * (pm->getRho(i) * dudrho[i]) * divV[i]); // CU1=0
        vec3<fptype> vNew; //, xNew;
        vNew = tmp.vel + dtH * (fptype(-1.0) * gradP(pid) / tmp.rho);

        tmp.u = uNew;
        tmp.vel = vNew;
        pm->setParticle(pid, tmp);
      } // end kokkos_lambda
    ); // end parallel_for
  } //endif(AFLAG)
} // end advanceParticlesHalfStep

void SPH::advanceParticles( const fptype dt ) {
  const fptype dtH = 0.5 * dt;

  // Advance the internal energies, velocities, and positions
  if (AFLAG == 0) {
    Kokkos::parallel_for("Triforce::SPH::advanceParticles::loop1",
      pm->allParticlesHostPolicy(),
      KOKKOS_LAMBDA(const uint32_t &pid) {
        Particle tmp = pm->getParticle(pid);
        fptype uNew;
        vec3<fptype> vNew, xNew;

        uNew = max(uMin(pid) + dtH * du(pid), 0.0);
        vNew = vMin(pid) + dtH * dv(pid);           // + av?
        xNew = tmp.loc + dt * vNew;

        tmp.u = uNew;
        tmp.vel = vNew;
        tmp.loc = xNew;
        pm->setParticle(pid, tmp);
      } // end kokkos_lambda
    ); // end parallel_for
  } else {
    Kokkos::parallel_for("Triforce::SPH::advanceParticles::loop2",
      pm->allParticlesHostPolicy(),
      KOKKOS_LAMBDA(const uint32_t &pid) {
        Particle tmp = pm->getParticle(pid);
        fptype uNew;
        vec3<fptype> vNew, xNew;
        uNew = tmp.u + dtH * (fptype(-1.0) * tmp.p / (PTINY + tmp.rho) * divV(pid));
        // + CU1 * (pm->getRho(i) * dudrho[i]) * divV[i]); // CU1=0
        vNew = tmp.vel + dtH * (fptype(-1.0) * gradP(pid) / (PTINY + tmp.rho) );
        xNew = tmp.loc + dt * vNew;

        tmp.u = uNew;
        tmp.vel = vNew;
        tmp.loc = xNew;
        pm->setParticle(pid, tmp);
      } // end kokkos_lambda
    ); // end parallel_for
  } // end if(AFLAG)
} // end advanceParticles

#endif // TRIFORCE_SPH_BASE_H
