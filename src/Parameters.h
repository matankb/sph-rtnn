#ifndef TRIFORCE_PARAMETERS_H
#define TRIFORCE_PARAMETERS_H


#include "Utilities.h"

enum ProblemType {SHEAR, DIFFUSION};

struct Simulation {
  std::string name;
  uint32_t smoothing_kernel;
  uint32_t smoothing_length_evolution;
  bool artificial_viscosity;
  bool artificial_heat;
  uint32_t save_step;
  uint32_t print_step;
  uint32_t max_neighbors;
  uint32_t neighborOption = 0;
  uint32_t np_total;
};


struct Domain {
  vec3<uint32_t> dims;
  vec3<uint32_t> nppc; // specify particles per cell to define mesh
  uint32_t nGhosts = 0;
  fptype x0, x1, y0, y1, z0, z1;  // physical dimensions
};


struct ParticleGroup {
  std::string name;
  vec3<uint32_t> num_particles;
  fptype mass_density;
  fptype internal_energy;
  fptype pressure;
  vec3<fptype> velocity;
  fptype gamma;
  Domain init_region;
};

struct Time {
  fptype t_final;
  fptype CFL;
  bool enableCFL = false;
  fptype dt;
  uint32_t Nt;
};

class Parameters {
public:
  explicit Parameters(ProblemType problem) {
    switch(problem) {
      case SHEAR:
        init_shear_problem();
        break;

      case DIFFUSION:
        init_diffusion_problem();
        break;

      default:
        exit(0);
    }
  }

  void init_shear_problem();
  void init_diffusion_problem();

public:
  Simulation sim;
  Domain mesh;
  Time time;
  ParticleGroup p1;
};

void Parameters::init_shear_problem() {
  // Simulation Parameters
  sim.name = "Shear Cavity";
  sim.smoothing_kernel = 1;
  sim.smoothing_length_evolution = 1;
  sim.artificial_viscosity = true;
  sim.artificial_heat = true;
  sim.save_step = 4;
  sim.print_step = 4;
  sim.max_neighbors = 100;
  sim.neighborOption = 0; // 0 = directFind (default), 1 = linkedList

  // Temporal Parameters
  time.dt = 5.0E-5; // seconds
  time.t_final = 0.15; // seconds
  time.Nt = static_cast<int>(time.t_final / time.dt) + 1; // Does this need to be rounded up/down?
  time.CFL = 0.9;

  // The only particle group
  p1.name = "fluid";
  p1.num_particles = {501, 1, 1};
  p1.mass_density = 1000.0;    // kg/m^3
  p1.internal_energy = 357.1;  // Joules
  p1.velocity = {0.0, 0.0, 0.0};

  // set total number of particles
  sim.np_total = p1.num_particles[0] * p1.num_particles[1] * p1.num_particles[2];

  // Particle Init Region (meters)
  p1.init_region.x0 = 0.0;
  p1.init_region.y0 = 0.0;
  p1.init_region.z0 = 0.0;
  p1.init_region.x1 = 1.0E-3;
  p1.init_region.y1 = 1.0E-3;
  p1.init_region.z1 = 0.0;

  // Chaining Mesh Parameters
  mesh.dims = {20, 20, 20};
  mesh.nGhosts = 2;
  mesh.x0 = -1.0;
  mesh.y0 = -1.0;
  mesh.z0 = -1.0;
  mesh.x1 = 1.0;
  mesh.y1 = 1.0;
  mesh.z1 = 1.0;
}

void Parameters::init_diffusion_problem() {
  sim.name = "Diffusion";
  sim.smoothing_kernel = 1;
  sim.smoothing_length_evolution = 1;
  sim.artificial_viscosity = false;
  sim.artificial_heat = false;
  sim.save_step = 1;
  sim.print_step = 1;
  sim.max_neighbors = 100;
  sim.neighborOption = 0; // 0 = directFind (default), 1 = linkedList

  time.dt = 5.0E-5; // seconds
  time.t_final = 0.1; // seconds
  time.Nt = static_cast<int>(time.t_final / time.dt) + 1;
  time.CFL = 0.1;
  time.enableCFL = false;

//    ParticleGroup p1;
  p1.name = "fluid";
  p1.num_particles = {10000, 1, 1}; 
  p1.mass_density = 1.0;    // kg/m^3
  p1.pressure = 1.0; // Pa
  p1.internal_energy = 2.5;  // Joules
  p1.velocity = {0.0, 0.0, 0.0};
  p1.gamma = 1.4;

  // meters
  p1.init_region.x0 = -0.5;
  p1.init_region.y0 = -0.5;
  p1.init_region.z0 = -0.5;
  p1.init_region.x1 = 0.5;
  p1.init_region.y1 = 0.5;
  p1.init_region.z1 = 0.5;

  mesh.dims = {10, 2, 2};
  mesh.nppc = {3, 1, 1};
  mesh.nGhosts = 0;
  mesh.x0 = -1.0;
  mesh.y0 = -1.0;
  mesh.z0 = -1.0;
  mesh.x1 = 1.0;
  mesh.y1 = 1.0;
  mesh.z1 = 1.0;
}

#endif //TRIFORCE_PARAMETERS_H
