#ifndef TRIFORCE_KERNELS_H
#define TRIFORCE_KERNELS_H
/*
 * Kernel.h
 *
 *
 *
 * Created On: 07/26/2022
 *
 * Last Updated:
 *    * MJL - 07/26/2022 - Initially created
 *         (Ref: https://pysph.readthedocs.io/en/latest/reference/kernels.html)
 */

#include "Utilities.h"

class Kernel {
public:
  // Constructors & Deconstructor
  explicit Kernel( uint32_t kernelType_input ) {
    kernelType = kernelType_input;
    scale_k = 1.0;

    if (kernelType == 1) {
        scale_k = 2.0;
    } else {
        scale_k = 3.0;
    }

    kernelType_d = Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space>("Kernel::kernelType_d");
    scale_k_d = Kokkos::View<fptype, Kokkos::DefaultExecutionSpace::memory_space>("Kernel::scale_k_d");

    Kokkos::deep_copy(kernelType_d, kernelType);
    Kokkos::deep_copy(scale_k_d, scale_k);
  };

  virtual ~Kernel() = default;

  // Methods
  KOKKOS_FUNCTION void computeKernel(fptype &w, vec3<fptype> &dwdx, vec3<fptype> dx, fptype radius, fptype hsml) const {
    if (kernelType == 1) {
        cubic(w, dwdx, dx, radius, hsml);
    } else if (kernelType == 2) {
        Gauss(w, dwdx, dx, radius, hsml);
    } else if (kernelType == 3) {
        quintic(w, dwdx, dx, radius, hsml);
    }
  };

  KOKKOS_FUNCTION void cubic(fptype &w, vec3<fptype> &dwdx, vec3<fptype> dx, fptype radius, fptype hsml) const;
  KOKKOS_FUNCTION void Gauss(fptype &w, vec3<fptype> &dwdx, vec3<fptype> dx, fptype radius, fptype hsml) const;
  KOKKOS_FUNCTION void quintic(fptype &w, vec3<fptype> &dwdx, vec3<fptype> dx, fptype radius, fptype hsml) const;

public: // Public Class Data
  uint32_t kernelType;
  fptype scale_k = 1.0;
  Kokkos::View<uint32_t, Kokkos::DefaultExecutionSpace::memory_space> kernelType_d;
  Kokkos::View<fptype, Kokkos::DefaultExecutionSpace::memory_space> scale_k_d;
};

KOKKOS_FUNCTION void Kernel::cubic(fptype &w, vec3<fptype> &dwdx, vec3<fptype> dx, fptype radius, fptype hsml) const {
  fptype factor = 1.0;
  if (DIM == 1) {
    factor = 1.0 / hsml;
  } else if (DIM == 2) {
    factor = 15.0 / (7.0 * fpPI * SQR(hsml));
  } else if (DIM == 3) {
    factor = 3.0 / (2.0 * fpPI * CUBE(hsml));
  }

  fptype qk = ( hsml != 0.0 ) ? radius / hsml : 0.0;

  if (qk >= 0.0 && qk <= 1.0 ) {
    w = factor * (2.0 / 3.0 - SQR(qk) + CUBE(qk) / 2.0);
    dwdx = dx * factor * fptype(-2.0 + 3.0 / 2.0 * qk) / SQR(hsml);
  } else if (qk > 1.0 && qk <= 2.0) {
    fptype two_minus_qk = 2.0 - qk;
    w = factor / 6.0 * CUBE(two_minus_qk);
    dwdx = dx / radius * fptype(-1.0 * factor * 3.0 / 6.0 * two_minus_qk * two_minus_qk / hsml);
  } else {
    w = 0.0;
    dwdx = {0.0, 0.0, 0.0};
  }
}

KOKKOS_FUNCTION void Kernel::Gauss(fptype &w, vec3<fptype> &dwdx, vec3<fptype> dx, fptype radius, fptype hsml) const {
  fptype qk = (hsml != 0.0) ? radius/ hsml : 0.0;
  fptype factor = 1.0 / (pow(hsml, fpDIM) * pow(fpPI, fpDIM / 2.0));

  if (qk >= 0.0 && qk <= 1.0e3) {
    w = factor * exp(-1.0 * SQR(qk));
    dwdx = dx * w * ((float) -2.0) / SQR(hsml);
  } else {
    w = 0.0;
    dwdx = {0.0, 0.0, 0.0};
  }
};

KOKKOS_FUNCTION void Kernel::quintic(fptype &w, vec3<fptype> &dwdx, vec3<fptype> dx, fptype radius, fptype hsml) const {
  fptype qk = (hsml != 0.0) ? radius/ hsml : 0.0;
  fptype factor = 1.0;

  if (DIM == 1) {
    factor = 1.0 / (120.0 * hsml);
  } else if (DIM == 2) {
    factor = 7.0 / (478.0 * fpPI * SQR(hsml));
  } else if (DIM == 3) {
    factor = 1.0 / (120.0 * fpPI * CUBE(hsml));
  }

  if (qk >= 0.0 && qk <= 1.0) {
    w = factor * (pow(3.0 - qk, 5.0) - 6.0 * pow(2.0 - qk, 5.0) + 15.0 * pow(1.0 - qk, 5.0));
    dwdx = dx * factor * fptype(-120.0 + 120.0 * qk - 50.0 * qk * qk) / SQR(hsml);
  } else if (qk > 1.0 && qk <= 2.0) {
    w = factor * (pow(3.0 - qk, 5.0) - 6.0 * pow(2.0 - qk, 5.0));
    dwdx = dx / radius * factor * fptype(-5.0 * pow(3.0 - qk, 4.0) + 30.0 * pow(2.0 - qk, 4.0)) / hsml;
  } else if (qk > 2.0 && qk < 3.0) {
    w = factor * pow(3.0 - qk, 5.0);
    dwdx = dx / radius * factor * fptype(-5.0 * pow(3.0 - qk, 4.0)) / hsml;
  } else {
    w = 0.0;
    dwdx = {0.0, 0.0, 0.0};
  }
}

#endif // TRIFORCE_KERNELS_H
