#include "Utilities.h"
#include <stdint.h>
#include <Kokkos_Core.hpp>

#include "IO.h"
#include "Parameters.h"
#include "ChainingMesh.h"
#include "LinkedList.h"
#include "SPH.h"

// #include "optix.h"

int main(int argc, char **argv) {

	/*
	int numPoints = 8;
	float *points = new float[3 * numPoints];

	points[0] = 0.0;
	points[1] = 0.0;
	points[2] = 0.0;
	points[3] = 9.72500038147;
	points[4] = 2.72699999809;
	points[5] = 73.8320007324;
	points[6] = 5.0;
	points[7] = 0.0;
	points[8] = 0.0;
	points[9] = 9.95300006866;
	points[10] = 2.6970000267;
	points[10] = 72.9100036621;
	points[11] = 6.0;
	points[12] = 0.0;
	points[13] = 0.0;
	points[15] = 9.91199970245;
	points[16] = 2.71600008011;
	points[17] = 73.466003418;
	points[18] = 4.99;
	points[19] = 4.99;
	points[20] = 0;
	points[21] = 9.56000041962;
	points[22] = 2.54600000381;
	points[23] = 68.4359970093;

	float** neighbors = getNeighborList(points, numPoints);

	printf("\n======= NOW PRINTING NEIGHBORS LIST ====== \n\n");
	// int i = 40;
	// printf("%.20f , %.20f, %d\n\n", neighbors[0][0], points[21], neighbors[0][0] == points[21]);
	for (int i = 0; i < (numPoints * numPoints); i++) {
		float* neighbor = neighbors[i];
		if (neighbor == NULL) {
			// printf("Neighbor is null!");
			// continue;
			return;
		}
		
		// get index of points
		int first_index = -1;
		int second_index = -1;
		for (int j = 0; j < numPoints; j++) {
			float x = points[(j * 3)];
			float y = points[(j * 3) + 1];
			float z = points[(j * 3) + 2];
			if (i == 0) {
				// printf("%0.5f, %0.5f, %0.5f\n", x, y, z);
			}
			if (neighbor[0] == x && neighbor[1] == y && neighbor[2] == z) {
				first_index = j;
			}
			if (neighbor[3] == x && neighbor[4] == y && neighbor[5] == z) {
				second_index = j;
			}
		}

		printf("<%f, %f, %f>, <%f, %f, %f>\n", neighbor[0], neighbor[1], neighbor[2], neighbor[3], neighbor[4], neighbor[5]);
		printf("  <%d, %d>\n", first_index, second_index);
	}

	// themain();

	// int* neighbors = getNeighborList(points, numPoints);
	// printf("\nThe numbers we got: %d, %d\n", neighbors[0], neighbors[1]);
	
	// testing(points, numPoints);
	// please();
	// printf("%d", testing());

	// int* points = (int*) calloc(3, sizeof(int));
	// points[0] = 12;
	// points[1] = 100;
	// points[2] = 23;
	// please();

	// printf("HERE IS THE NUMBER: %d <----\n", please());
	// please();
	// printf("HERE IS ANOTHER NUMBER %d <--- should be seven", generate_state()->numQueries);
	// please()

	*/

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

