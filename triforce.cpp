#include "Utilities.h"
#include <Kokkos_Core.hpp>
#include <math.h>

#include "IO.h"
#include "Parameters.h"
#include "ChainingMesh.h"
#include "LinkedList.h"
#include "SPH.h"

void example() {
	fptype radius_limit = 0.040404;
	fptype p1_x = 0.045454561710357666015625;
	fptype p2_x = 0.00505056045949459075927734375;

	// test using manual method
	vec3<fptype> loc1{p1_x, -0.500000, -0.500000};
	vec3<fptype> loc2{p2_x, -0.500000, -0.500000};
	vec3<fptype> vel{0, 0, 0};
	Particle pi(1, 1, 1, 1, 1, 1, loc1, vel);
	Particle pj(1, 1, 1, 1, 1, 1, loc2, vel);

	vec3<fptype> dwdx;
	vec3<fptype> dx = pi.loc - pj.loc;
	fptype radius = dx.length();
	printf("%0.70f\n", radius);
	float square = (p1_x - p2_x) * (p1_x - p2_x);
	printf("%0.70f, %0.70f\n", square, radius_limit * radius_limit);
	bool finds_neighbor_manual = radius <= radius_limit;

	// test using rtnn
	fptype* points = new fptype[3 * 3];
	points[0] = p1_x;
	points[1] = -0.5;
	points[2] = -0.5;
	points[3] = p2_x;
	points[4] = -0.5;
	points[5] = -0.5;

	return;

	/*
	fptype** neighbors = getNeighborList(points, 2, radius_limit, 0);
	int pairs_generated_rtnn = 0;
	int i = 0;
	while (neighbors[i] != NULL) {
		if (neighbors[i][0] == neighbors[i][3] && neighbors[i][1] == neighbors[i][4] && neighbors[i][2] == neighbors[i][5]) {
			i++;
			continue; // since `points` is both the query and search array, rtnn will return the same (p, p) as a neighbor pair, but we don't want to count that
		}
		pairs_generated_rtnn++;
		printf("== Neighbor Pair ==\n");
		for (int j = 0; j < 6; j++) {
			printf("%0.70f\n", neighbors[i][j]);
		}
		i++;
	}

	printf("\nInside radius for RTNN? %d\n", pairs_generated_rtnn > 0);
	printf("Inside radius for SPH? %d\n", finds_neighbor_manual);
	*/
}

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

			sim.step( params.time.dt, tt );
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
