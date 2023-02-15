#include "Utilities.h"
#include <Kokkos_Core.hpp>

#include "IO.h"
#include "Parameters.h"
#include "ChainingMesh.h"
#include "LinkedList.h"
#include "SPH.h"

int main(int argc, char **argv) {

	Kokkos::initialize( argc, argv );
	{
		std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
		std::cout << " TFLuid(?) \n";
		std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

		auto startRun = high_resolution_clock::now();

		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// Initialize Simulation
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		auto startInit = high_resolution_clock::now();

		std::cout << "~~~ Initializing...\n";

		Parameters params(DIFFUSION);
		SPH sim(&params);

		save_particles_to_csv(*sim.pm, sim.getSimName(), 0);

		std::cout << "~~~ Initialization complete.\n";

		auto endInit = high_resolution_clock::now();

		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// Integration
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		auto startIntegration = high_resolution_clock::now();

		std::cout << "~~~ Beginning simulation...\n";

		fptype currentTime = 0.0;
		uint32_t Nt = params.time.Nt;

		// ~~~~~ First step (sets up leapfrog) ~~~~~
		sim.report_start(1, currentTime);

		sim.initStep( params.time.dt / 2.0 );
		currentTime += params.time.dt;

		sim.report_end(1);

		if ( params.sim.save_step == 1 ) {
			save_particles_to_csv(*sim.pm, sim.getSimName(), 1);
		}

		// ~~~~~ Main Loop ~~~~~
		for (uint32_t tt = 2; tt <= Nt; tt++) {

			if ( tt % params.sim.print_step == 0)  {
				sim.report_start(tt, currentTime);
			}

			sim.step( params.time.dt );
			currentTime += params.time.dt;

			if ( tt % params.sim.print_step == 0 ) {
				sim.report_end(tt);
			}

			if ( tt % params.sim.save_step == 0 ) {
				save_particles_to_csv(*sim.pm, sim.getSimName(), tt);
			}

		} // end for(tt)

		std::cout << "~~~ Simulation complete.\n";
		auto endIntegration = high_resolution_clock::now();


		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// Finalize
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		auto endRun = high_resolution_clock::now();

		std::cout << "\n~~~ Simulation runtime:\n";
		double runTime = (duration_cast<microseconds>(endRun - startRun)).count();
		double initTime = (duration_cast<microseconds>(endInit - startInit)).count();
		double integrationTime = (duration_cast<microseconds>(endIntegration - startIntegration)).count();

		std::cout << "\n     Total run-time:      " << 1e-6 * runTime << " s";
		std::cout << "\n     Initialization:      " << 1e-6 * initTime << " s";
		std::cout << "\n        Integration:      " << 1e-6 * integrationTime << " s";
		std::cout << "\n      Per 100 steps:      " << 1e-6 * runTime / params.time.Nt * 100 << " s";


		std::cout << "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
		std::cout << " end ";
		std::cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";
	}
	Kokkos::finalize();

	return 0;
}
