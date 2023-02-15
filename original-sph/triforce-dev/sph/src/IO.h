#ifndef TRIFORCE_IO_H
#define TRIFORCE_IO_H

#include <filesystem>

#include "Utilities.h"
#include "ParticleManagerBase.h"

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

#endif // TRIFORCE_IO_H
