#ifndef TRIFORCE_IO_H
#define TRIFORCE_IO_H

#include <filesystem>

#include "Utilities.h"
#include "ParticleManagerBase.h"
#include "NeighborsBase.h"

namespace fs = std::filesystem;

void save_particles_to_csv(const ParticleManagerBase &pm, const std::string &sim_name, const uint32_t timestep) {
  std::cout << " Saving frame number " << timestep << std::endl;
  // Set path to .../triforce/sph/data/simulation_name
  // If directory does not exist, it is created.
  // fs::creat_directory return 1 if created, 0 if already exists (or is not created)
  fs::path path = fs::current_path();
  path /= fs::path("data/" + sim_name);
  fs::create_directory(path);

  // Create file path, open ofstream object
  fs::path filename(path.string() + "/particles_" + std::to_string(timestep) + ".csv");
  std::ofstream file(filename);

  // Header line
  file << "x, y, z, vx, vy, vz, mass, rho, pressure, internal energy, sound speed, hsml\n";

  // Set precision to max
  file << std::setprecision(std::numeric_limits<fptype>::max_digits10);

  // Iterate over particles and print to file
  for (uint32_t i = 0; i < pm.pNum; i++) {
    file << pm.getParticle(i) << "\n";
  }

}

void save_neighbor_list_to_csv(
  const Kokkos::View<neighbors*>::HostMirror neighborList, 
  const ParticleManagerBase &pm,
  Kokkos::View<uint32_t>::HostMirror &pairCounter, 
  const const uint32_t timestep
) {
  std::cout << " Saving neighborList number " << timestep << std::endl;

  // Set path to .../triforce/sph/data/simulation_name
  // If directory does not exist, it is created.
  // fs::creat_directory return 1 if created, 0 if already exists (or is not created)
  fs::path path = fs::current_path() / fs::path("data/neighbors-pid");
  fs::create_directory(path);

  // Create file path, open ofstream object
  fs::path filename(path.string() + "/neighbors" + std::to_string(timestep) + ".csv");
  std::ofstream file(filename);

  // Header line
  // file << "x1, y1, z1, x2, y2, z2\n";
  file << "pid1, pid2";

  // Set precision to max
  file << std::setprecision(std::numeric_limits<fptype>::max_digits10);

  // print out calculated updated neighborList sequentially
  for (uint32_t i = 0; i < pairCounter(); i++) {
    const neighbors n = neighborList(i);
    Particle p1 = pm.getParticleAtomic(n.pid1);
    Particle p2 = pm.getParticleAtomic(n.pid2);

    file << p1.loc.x() << "," << p1.loc.y()  << "," << p1.loc.z()  << ",";
    file << p2.loc.x() << "," << p2.loc.y()  << "," << p2.loc.z()  << "\n";
    // file << n.pid1 << "," << n.pid2 << "\n";
  }
}

#endif // TRIFORCE_IO_H
