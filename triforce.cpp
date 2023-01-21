#include "Utilities.h"
#include <Kokkos_Core.hpp>
#include <math.h>

#include "IO.h"
#include "Parameters.h"
#include "ChainingMesh.h"
#include "LinkedList.h"
#include "SPH.h"

void example(double radius_limit, double x) {
	// fptype mass, fptype rho,
    //                        fptype p, fptype u,
    //                        fptype c, fptype hsml,
    //                        vec3<fptype> loc, vec3<fptype> vel
	vec3<fptype> loc1{0.04545456171035766601562500000000000000000000000000000000000000000000000000000000000, -0.500000, -0.500000};
	vec3<fptype> loc2{0.0050505604594945907592773437500000000000000000000000000000000000000000000000000000000000000000000000, -0.500000, -0.500000};
	vec3<fptype> vel{0, 0, 0};
	Particle pi(1, 1, 1, 1, 1, 1, loc1, vel);
	Particle pj(1, 1, 1, 1, 1, 1, loc2, vel);


	// vec3<fptype> dwdx;
	vec3<fptype> dx = pi.loc - pj.loc;
	fptype radius = dx.length();

	printf("Radius              : %0.100f\n", radius);
	printf("radius squared      : %0.100f\n", radius * radius);
	// printf("random number:      : %0.100f\n", x);
	// printf("random number square: %0.100f | %s\n", x * x, radius <= x ? "yes" : "no");
	printf("radius limit        : %0.100f\n", radius_limit);
	printf("radius limit squared: %0.100f\n", radius_limit * radius_limit);

	double radius_rtnn = calculate_radius(loc1.x(), loc1.y(), loc1.z(), loc2.x(), loc2.y(), loc2.z());
	printf("Radius RTNN         : %0.100f\n", radius_rtnn);

	printf("\n\Inside radius for SPH? %d\n", radius <= radius_limit);
	printf("\n\Inside radius for RTNN? %d\n", radius_rtnn <= (radius_limit * radius_limit));
}

int main(int argc, char **argv) {
	// example(0.040404, 1);
	// return 0;
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
