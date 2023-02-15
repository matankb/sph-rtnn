#ifndef TRIFORCE_SIMULATION_BASE_H
#define TRIFORCE_SIMULATION_BASE_H
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
#include "Parameters.h"

class SimulationBase {
public:
  // Constructors & Deconstructor
  explicit SimulationBase( Parameters* params_input ) : params(params_input) {};
  virtual ~SimulationBase() = default;

public: // Class Functions
  virtual void step( const fptype dt ) {};
  virtual void report_start( uint32_t step, fptype time ) {};
  virtual void report_end( uint32_t step ) {};

  std::string getSimName() { return params->sim.name; }

protected: // Protected Class Data
  Parameters* params;
};

#endif // TRIFORCE_SIMULATION_BASE_H
