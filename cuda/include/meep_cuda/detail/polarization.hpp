#ifndef MEEP_CUDA_DETAIL_POLARIZATION_HPP
#define MEEP_CUDA_DETAIL_POLARIZATION_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define MEEP_CUDA_POLARIZATION_HOST_DEVICE __host__ __device__
#else
#define MEEP_CUDA_POLARIZATION_HOST_DEVICE
#endif

namespace meep_cuda {
namespace detail {

struct lorentzian_operands_fp32 {
  const float *w1;
  const float *w2;
  const float *sigma1;
  const float *sigma2;
  std::ptrdiff_t stride1;
  std::ptrdiff_t stride2;
};

MEEP_CUDA_POLARIZATION_HOST_DEVICE inline lorentzian_operands_fp32
normalize_lorentzian_operands(const float *w1, const float *w2,
                              const float *sigma1, const float *sigma2,
                              std::ptrdiff_t stride1,
                              std::ptrdiff_t stride2) {
  if (sigma2 && !sigma1)
    return {w2, w1, sigma2, sigma1, stride2, stride1};
  return {w1, w2, sigma1, sigma2, stride1, stride2};
}

MEEP_CUDA_POLARIZATION_HOST_DEVICE inline float
lorentzian_offdiagonal_average_fp32(const float *coefficient,
                                    const float *field,
                                    std::ptrdiff_t index,
                                    std::ptrdiff_t field_stride,
                                    std::ptrdiff_t neighbor_stride) {
  return 0.25f *
         ((field[index] + field[index - neighbor_stride]) *
              coefficient[index] +
          (field[index + field_stride] +
           field[index + field_stride - neighbor_stride]) *
              coefficient[index + field_stride]);
}

struct lorentzian_coefficients_fp32 {
  float gamma_inverse;
  float gamma_previous;
  float omega_dt_squared;
  float omega_dt_squared_denominator;
};

MEEP_CUDA_POLARIZATION_HOST_DEVICE inline bool
lorentzian_positive_zero_fp32(float value) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(value) == 0U;
#else
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits == 0U;
#endif
}

MEEP_CUDA_POLARIZATION_HOST_DEVICE inline bool
lorentzian_sparse_point_active_fp32(float polarization, float auxiliary,
                                    float sigma) {
  return !lorentzian_positive_zero_fp32(polarization) ||
         !lorentzian_positive_zero_fp32(auxiliary) ||
         !lorentzian_positive_zero_fp32(sigma);
}

MEEP_CUDA_POLARIZATION_HOST_DEVICE inline bool
lorentzian_finite_fp32(float value) {
#if defined(__CUDA_ARCH__)
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

MEEP_CUDA_POLARIZATION_HOST_DEVICE inline bool
lorentzian_coefficients_finite_fp32(
    const lorentzian_coefficients_fp32 &coefficients) {
  return lorentzian_finite_fp32(coefficients.gamma_inverse) &&
         lorentzian_finite_fp32(coefficients.gamma_previous) &&
         lorentzian_finite_fp32(coefficients.omega_dt_squared) &&
         lorentzian_finite_fp32(
             coefficients.omega_dt_squared_denominator);
}

MEEP_CUDA_POLARIZATION_HOST_DEVICE inline void
apply_lorentzian_update_fp32(
    float &polarization, float &previous_polarization, float field,
    float sigma, float offdiagonal1, float offdiagonal2,
    bool anisotropic,
    const lorentzian_coefficients_fp32 &coefficients,
    bool elide_bit_identical_positive_zero = false) {
  // Match Meep's boundary-stability guard for anisotropic sigma exactly.
  if (anisotropic && sigma == 0.0f) return;

  const float current = polarization;
  const float next_polarization =
      coefficients.gamma_inverse *
      (current *
           (2.0f - coefficients.omega_dt_squared_denominator) -
       coefficients.gamma_previous * previous_polarization +
       coefficients.omega_dt_squared *
           (sigma * field + offdiagonal1 + offdiagonal2));
  if (elide_bit_identical_positive_zero && !anisotropic &&
      lorentzian_positive_zero_fp32(current) &&
      lorentzian_positive_zero_fp32(previous_polarization) &&
      lorentzian_positive_zero_fp32(sigma) &&
      lorentzian_finite_fp32(field) &&
      lorentzian_coefficients_finite_fp32(coefficients) &&
      lorentzian_positive_zero_fp32(next_polarization))
    return;
  polarization = next_polarization;
  previous_polarization = current;
}

// Algebraically equivalent Lorentzian/Drude recurrence using the current
// polarization and its most recent increment as state.  Keeping the
// increment explicitly avoids recovering it by subtracting two nearly equal
// FP32 polarization values on every timestep.
MEEP_CUDA_POLARIZATION_HOST_DEVICE inline void
apply_lorentzian_increment_update_fp32(
    float &polarization, float &polarization_increment, float field,
    float sigma, float offdiagonal1, float offdiagonal2,
    bool anisotropic,
    const lorentzian_coefficients_fp32 &coefficients,
    bool elide_bit_identical_positive_zero = false) {
  // Match Meep's boundary-stability guard for anisotropic sigma exactly.
  if (anisotropic && sigma == 0.0f) return;

  const float next_increment =
      coefficients.gamma_inverse *
      (coefficients.gamma_previous * polarization_increment -
       coefficients.omega_dt_squared_denominator * polarization +
       coefficients.omega_dt_squared *
           (sigma * field + offdiagonal1 + offdiagonal2));
  if (elide_bit_identical_positive_zero && !anisotropic &&
      lorentzian_positive_zero_fp32(polarization) &&
      lorentzian_positive_zero_fp32(polarization_increment) &&
      lorentzian_positive_zero_fp32(sigma) &&
      lorentzian_finite_fp32(field) &&
      lorentzian_coefficients_finite_fp32(coefficients) &&
      lorentzian_positive_zero_fp32(next_increment) &&
      lorentzian_positive_zero_fp32(polarization + next_increment))
    return;
  polarization += next_increment;
  polarization_increment = next_increment;
}

// A sparse plan may omit a positive-zero state only when the exact recurrence
// preserves positive zero for every finite field sign class.  This predicate
// deliberately executes the same recurrence helpers as the update kernels;
// it is compiled for both host and device so validation can require agreement.
MEEP_CUDA_POLARIZATION_HOST_DEVICE inline bool
lorentzian_sparse_coefficients_safe_fp32(
    const lorentzian_coefficients_fp32 &coefficients,
    bool increment_state) {
  if (!lorentzian_coefficients_finite_fp32(coefficients)) return false;
  const float field_sign_classes[4] = {0.0f, -0.0f, 1.0f, -1.0f};
  for (int field_class = 0; field_class < 4; ++field_class) {
    float polarization = 0.0f;
    float auxiliary = 0.0f;
    if (increment_state)
      apply_lorentzian_increment_update_fp32(
          polarization, auxiliary, field_sign_classes[field_class], 0.0f,
          0.0f, 0.0f, false, coefficients, false);
    else
      apply_lorentzian_update_fp32(
          polarization, auxiliary, field_sign_classes[field_class], 0.0f,
          0.0f, 0.0f, false, coefficients, false);
    if (!lorentzian_positive_zero_fp32(polarization) ||
        !lorentzian_positive_zero_fp32(auxiliary))
      return false;
  }
  return true;
}

enum gyrotropic_model_fp32 {
  gyrotropic_lorentzian_fp32 = 0,
  gyrotropic_drude_fp32 = 1,
  gyrotropic_saturated_fp32 = 2
};

struct gyrotropic_coefficients_fp32 {
  int model;
  float inverse[9];
  float gyro[9];
  float diagonal;
  float gamma_previous;
  float omega_dt_squared;
  float precession_half_dt;
  float omega_dt;
  float gamma_dt;
  float alpha_half;
  float drive_dt;
};

MEEP_CUDA_POLARIZATION_HOST_DEVICE inline float
gyrotropic_offdiagonal_average_fp32(
    const float *field, std::ptrdiff_t index,
    std::ptrdiff_t field_stride, std::ptrdiff_t neighbor_stride) {
  return field
             ? 0.25f *
                   (field[index] + field[index - neighbor_stride] +
                    field[index + field_stride] +
                    field[index + field_stride - neighbor_stride])
             : 0.0f;
}

MEEP_CUDA_POLARIZATION_HOST_DEVICE inline void
apply_gyrotropic_update_fp32(
    float &polarization0, float &polarization1, float &polarization2,
    float &previous0, float &previous1, float &previous2,
    float field0, float field1, float field2, float sigma,
    const gyrotropic_coefficients_fp32 &coefficients) {
  const float p0 = polarization0;
  const float p1 = polarization1;
  const float p2 = polarization2;
  float r0;
  float r1;
  float r2;

  if (coefficients.model != gyrotropic_saturated_fp32) {
    r0 = coefficients.diagonal * p0 -
         coefficients.gamma_previous * previous0 +
         coefficients.omega_dt_squared * sigma * field0 -
         coefficients.precession_half_dt * coefficients.gyro[1] *
             previous1 -
         coefficients.precession_half_dt * coefficients.gyro[2] *
             previous2;
    r1 = coefficients.diagonal * p1 -
         coefficients.gamma_previous * previous1 +
         coefficients.omega_dt_squared * sigma * field1 -
         coefficients.precession_half_dt * coefficients.gyro[3] *
             previous0 -
         coefficients.precession_half_dt * coefficients.gyro[5] *
             previous2;
    r2 = coefficients.diagonal * p2 -
         coefficients.gamma_previous * previous2 +
         coefficients.omega_dt_squared * sigma * field2 -
         coefficients.precession_half_dt * coefficients.gyro[7] *
             previous1 -
         coefficients.precession_half_dt * coefficients.gyro[6] *
             previous0;
  }
  else {
    const float q0 =
        -coefficients.omega_dt * p0 +
        coefficients.alpha_half * previous0 +
        coefficients.drive_dt * sigma * field0;
    const float q1 =
        -coefficients.omega_dt * p1 +
        coefficients.alpha_half * previous1 +
        coefficients.drive_dt * sigma * field1;
    const float q2 =
        -coefficients.omega_dt * p2 +
        coefficients.alpha_half * previous2 +
        coefficients.drive_dt * sigma * field2;
    r0 = 0.5f * previous0 - coefficients.gamma_dt * p0 +
         coefficients.gyro[1] * q1 + coefficients.gyro[2] * q2;
    r1 = 0.5f * previous1 - coefficients.gamma_dt * p1 +
         coefficients.gyro[5] * q2 + coefficients.gyro[3] * q0;
    r2 = 0.5f * previous2 - coefficients.gamma_dt * p2 +
         coefficients.gyro[6] * q0 + coefficients.gyro[7] * q1;
  }

  previous0 = p0;
  previous1 = p1;
  previous2 = p2;
  polarization0 = coefficients.inverse[0] * r0 +
                  coefficients.inverse[1] * r1 +
                  coefficients.inverse[2] * r2;
  polarization1 = coefficients.inverse[3] * r0 +
                  coefficients.inverse[4] * r1 +
                  coefficients.inverse[5] * r2;
  polarization2 = coefficients.inverse[6] * r0 +
                  coefficients.inverse[7] * r1 +
                  coefficients.inverse[8] * r2;
}

} // namespace detail
} // namespace meep_cuda

#undef MEEP_CUDA_POLARIZATION_HOST_DEVICE

#endif
