#ifndef TRIFORCE_UTILITIES_H
#define TRIFORCE_UTILITIES_H
/*
 * Utilities.h
 *
 * Utilities defines routines used in adaptive
 * these include
 *
 *
 *
 * Created On: 07/18/2022
 *
 * Last Modified:
 *    * MJL - 07/19/2022 - Added Utilities file with c libs
 *    * MJL - 07/19/2022 - added new template functions
 *    * AJK = 07/19/2022 - fixed chrono namespace
 *    * ATS - 07/19/2022 - removed math funcs (use stdlib), changed typedef->using,
 *                         added NaN check macro, added vec3 and uint3 structs
 *    * RLM - 07/25/2022 - added algorithm package so that sort works
 *    * ATS - 07/28/2022 - organized includes, added filesystem include
 */

// Type includes
#include <cstdint>
#include <vector>
#include <string>

// IO includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>

// Math includes
#include <cmath>

// Utility includes
#include <cassert>
#include <chrono>
#include <random>
#include <algorithm>
#include <memory>

// Kokkos includes
#include <Kokkos_Core.hpp>
//#include <Kokkos_MathematicalFunctions.hpp>

// Project includes
#include "Vector3.h"

using namespace std::chrono;
using std::unique_ptr;
using std::make_unique;
using std::shared_ptr;
using std::make_shared;

#define USE_FLOATS 1

// ---- data type for real numbers -----
#ifdef USE_FLOATS
using fptype = float;
#else
using fptype = double;
#endif

// Aliases
using fp_array = unique_ptr<fptype[]>;
using fpvec_array = unique_ptr<vec3<fptype>[]>;
using ui_array = unique_ptr<uint32_t[]>;
using uivec_array = unique_ptr<vec3<uint32_t>[]>;

// Constants
constexpr uint32_t DIM = 1;
constexpr fptype fpDIM = static_cast<fptype>(DIM);
constexpr fptype fpPI = static_cast<fptype>(M_PI);
constexpr fptype PTINY = std::numeric_limits<fptype>::min() * 10.0; // how to get
constexpr fptype PLOW = std::numeric_limits<fptype>::lowest();
constexpr fptype PHIGH = std::numeric_limits<fptype>::max();
constexpr uint32_t AFLAG = 1;

// Utility Functions
#define CheckNAN(ans, file, line) { nanAssert((ans), file, line); }
inline void nanAssert(fptype val, const char *file, int line, bool abort = true) {
  if (std::isnan(val)) {
    std::cout << file << "[" << line << "]: NaN value encountered." << std::endl;
    if (abort) { exit(-1); }
  }
}

template <class T>
KOKKOS_INLINE_FUNCTION T SQR(const T a) {return a * a;}

template <class T>
KOKKOS_INLINE_FUNCTION T CUBE(const T a) {return a * a * a;}

#endif // TRIFORCE_UTILITIES_H
