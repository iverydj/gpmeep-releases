#include "meep_cuda/runtime.hpp"
#include "meep_cuda/detail/polarization.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct device_deleter {
  void operator()(void *pointer) const { meep_cuda::free_device(pointer); }
};
using device_pointer = std::unique_ptr<void, device_deleter>;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T>
device_pointer copy_to_device(const std::vector<T> &values) {
  if (values.empty()) return device_pointer(nullptr);
  const std::size_t bytes = values.size() * sizeof(T);
  device_pointer result(meep_cuda::allocate_device_bytes(bytes));
  meep_cuda::copy_to_device(result.get(), values.data(), bytes);
  return result;
}

template <typename T>
device_pointer copy_scalar_to_device(const T &value) {
  device_pointer result(meep_cuda::allocate_device_bytes(sizeof(T)));
  meep_cuda::copy_to_device(result.get(), &value, sizeof(T));
  return result;
}

template <typename T>
std::vector<T> copy_from_device(const void *pointer, std::size_t count) {
  std::vector<T> result(count);
  if (count)
    meep_cuda::copy_to_host(result.data(), pointer, count * sizeof(T));
  return result;
}

template <typename T>
T copy_scalar_from_device(const void *pointer) {
  T result{};
  meep_cuda::copy_to_host(&result, pointer, sizeof(T));
  return result;
}

int selected_test_device() {
  std::string diagnostic;
  if (!meep_cuda::runtime_available(&diagnostic)) return -1;
  const std::vector<meep_cuda::device_info> devices =
      meep_cuda::enumerate_devices();
  const char *requested = std::getenv("MEEP_GPU_DEVICE");
  if (requested && requested[0]) {
    char *end = nullptr;
    const long ordinal = std::strtol(requested, &end, 10);
    if (!end || *end != '\0' || ordinal < 0 ||
        ordinal > std::numeric_limits<int>::max())
      throw std::invalid_argument(
          "MEEP_GPU_DEVICE must be a nonnegative integer");
    for (const auto &device : devices)
      if (device.ordinal == ordinal && meep_cuda::device_compatible(device))
        return device.ordinal;
    return -1;
  }
  for (const auto &device : devices)
    if (meep_cuda::device_compatible(device)) return device.ordinal;
  return -1;
}

meep_cuda::index_space_fp32 irregular_space() {
  meep_cuda::index_space_fp32 result{};
  result.field_start = 7;
  result.extent1 = 3;
  result.extent2 = 5;
  result.extent3 = 70;
  result.field_stride1 = 500;
  result.field_stride2 = 80;
  result.field_stride3 = 1;
  result.coefficient_start = -1;
  result.coefficient2_start = -1;
  return result;
}

std::vector<std::ptrdiff_t> decoded_indices(
    const meep_cuda::index_space_fp32 &space, std::size_t count) {
  std::vector<std::ptrdiff_t> result(count);
  for (std::size_t point = 0; point < count; ++point)
    result[point] =
        meep_cuda::detail::decode_index_space_fp32(space, point).field;
  return result;
}

void require_bitwise_equal(const std::vector<float> &actual,
                           const std::vector<float> &expected,
                           const char *label) {
  require(actual.size() == expected.size(),
          std::string(label) + " size mismatch");
  for (std::size_t index = 0; index < actual.size(); ++index)
    if (std::memcmp(&actual[index], &expected[index], sizeof(float)) != 0)
      throw std::runtime_error(
          std::string(label) + " differs at scalar " +
          std::to_string(index));
}

void check_compaction_fixture(const meep_cuda::index_space_fp32 &space,
                              std::size_t array_count,
                              bool increment_state,
                              const char *label) {
  const std::size_t point_count =
      space.extent1 * space.extent2 * space.extent3;
  const std::vector<std::ptrdiff_t> owned =
      decoded_indices(space, point_count);
  std::vector<float> polarization(array_count, 0.0f);
  std::vector<float> auxiliary(array_count, 0.0f);
  std::vector<float> sigma(array_count, 0.0f);
  for (std::size_t point = 0; point < point_count; ++point) {
    require(owned[point] >= 0 &&
                static_cast<std::size_t>(owned[point]) < array_count,
            std::string(label) + " index outside fixture allocation");
    const std::size_t index = static_cast<std::size_t>(owned[point]);
    if (point % 7 == 0) polarization[index] = 0.25f;
    if (point % 11 == 0) auxiliary[index] = -0.0f;
    if (point % 13 == 0) sigma[index] = 0.125f;
  }

  std::vector<std::ptrdiff_t> expected;
  for (const std::ptrdiff_t owned_index : owned) {
    const std::size_t index = static_cast<std::size_t>(owned_index);
    if (meep_cuda::detail::lorentzian_sparse_point_active_fp32(
            polarization[index], auxiliary[index], sigma[index]))
      expected.push_back(owned_index);
  }

  device_pointer device_p = copy_to_device(polarization);
  device_pointer device_aux = copy_to_device(auxiliary);
  device_pointer device_sigma = copy_to_device(sigma);
  meep_cuda::lorentzian_material_fp32 material{};
  material.sigma = static_cast<const float *>(device_sigma.get());
  material.gamma_inverse = 0.91f;
  material.gamma_previous = 0.87f;
  material.omega_dt_squared = 0.12f;
  material.omega_dt_squared_denominator = 0.12f;
  const std::size_t block_count =
      meep_cuda::lorentzian_sparse_compaction_block_count(point_count);
  std::vector<std::uint32_t> zero_counts(block_count, 0);
  device_pointer device_counts = copy_to_device(zero_counts);
  const std::uint32_t zero = 0;
  device_pointer device_validation = copy_scalar_to_device(zero);
  meep_cuda::count_lorentzian_sparse_active_blocks_fp32(
      static_cast<const float *>(device_p.get()),
      static_cast<const float *>(device_aux.get()), space, point_count,
      material, increment_state,
      static_cast<std::uint32_t *>(device_counts.get()),
      static_cast<std::uint32_t *>(device_validation.get()));
  const std::vector<std::uint32_t> counts =
      copy_from_device<std::uint32_t>(device_counts.get(), block_count);
  std::vector<std::uint64_t> offsets(block_count, 0);
  std::uint64_t active_count = 0;
  for (std::size_t block = 0; block < block_count; ++block) {
    offsets[block] = active_count;
    active_count += counts[block];
  }
  require(active_count == expected.size(),
          std::string(label) + " active count mismatch");
  device_pointer device_offsets = copy_to_device(offsets);
  std::vector<std::ptrdiff_t> output(expected.size(), -1);
  device_pointer device_output = copy_to_device(output);
  meep_cuda::fill_lorentzian_sparse_active_indices_fp32(
      static_cast<const float *>(device_p.get()),
      static_cast<const float *>(device_aux.get()), space, point_count,
      material, increment_state,
      static_cast<const std::uint64_t *>(device_offsets.get()),
      expected.size(), static_cast<std::ptrdiff_t *>(device_output.get()),
      static_cast<std::uint32_t *>(device_validation.get()));
  meep_cuda::synchronize();
  require(copy_scalar_from_device<std::uint32_t>(device_validation.get()) == 0,
          std::string(label) + " raised validation flag");
  require(copy_from_device<std::ptrdiff_t>(
              device_output.get(), expected.size()) == expected,
          std::string(label) + " did not preserve structured order");
}

void check_compaction_boundaries_and_2d_ordering() {
  const std::size_t boundary_counts[] = {255, 256, 257};
  for (const std::size_t point_count : boundary_counts) {
    meep_cuda::index_space_fp32 linear{};
    linear.extent1 = 1;
    linear.extent2 = 1;
    linear.extent3 = point_count;
    linear.field_stride3 = 1;
    linear.coefficient_start = -1;
    linear.coefficient2_start = -1;
    for (int state = 0; state < 2; ++state)
      check_compaction_fixture(
          linear, point_count, state != 0,
          point_count == 255
              ? "255-point compaction"
              : point_count == 256 ? "256-point compaction"
                                   : "257-point compaction");
  }

  meep_cuda::index_space_fp32 descending_2d{};
  descending_2d.field_start = 600;
  descending_2d.extent1 = 1;
  descending_2d.extent2 = 4;
  descending_2d.extent3 = 65;
  descending_2d.field_stride2 = -100;
  descending_2d.field_stride3 = -1;
  descending_2d.coefficient_start = -1;
  descending_2d.coefficient2_start = -1;
  const std::vector<std::ptrdiff_t> order = decoded_indices(
      descending_2d,
      descending_2d.extent2 * descending_2d.extent3);
  require(!std::is_sorted(order.begin(), order.end()),
          "2D fixture must exercise non-monotone structured order");
  for (int state = 0; state < 2; ++state)
    check_compaction_fixture(
        descending_2d, 700, state != 0,
        "non-monotone 2D compaction");
}

void check_device_coefficient_matrix() {
  meep_cuda::index_space_fp32 space{};
  space.extent1 = 1;
  space.extent2 = 1;
  space.extent3 = 1;
  space.coefficient_start = -1;
  space.coefficient2_start = -1;
  const std::vector<float> zero_state(1, 0.0f);
  device_pointer device_zero = copy_to_device(zero_state);
  const std::uint32_t zero = 0;
  device_pointer device_count = copy_scalar_to_device(zero);
  device_pointer device_validation = copy_scalar_to_device(zero);

  const float values[] = {
      0.0f,
      -0.0f,
      0.5f,
      -0.5f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  for (int state = 0; state < 2; ++state)
    for (int coefficient_index = 0; coefficient_index < 4;
         ++coefficient_index)
      for (const float value : values) {
        meep_cuda::lorentzian_material_fp32 material{};
        material.sigma = static_cast<const float *>(device_zero.get());
        material.gamma_inverse = 0.91f;
        material.gamma_previous = 0.87f;
        material.omega_dt_squared = 0.12f;
        material.omega_dt_squared_denominator = 0.12f;
        float *coefficients[] = {
            &material.gamma_inverse,
            &material.gamma_previous,
            &material.omega_dt_squared,
            &material.omega_dt_squared_denominator};
        *coefficients[coefficient_index] = value;
        const meep_cuda::detail::lorentzian_coefficients_fp32 host_tuple = {
            material.gamma_inverse,
            material.gamma_previous,
            material.omega_dt_squared,
            material.omega_dt_squared_denominator};
        const bool host_safe =
            meep_cuda::detail::lorentzian_sparse_coefficients_safe_fp32(
                host_tuple, state != 0);
        meep_cuda::copy_to_device(device_count.get(), &zero, sizeof(zero));
        meep_cuda::copy_to_device(
            device_validation.get(), &zero, sizeof(zero));
        meep_cuda::count_lorentzian_sparse_active_blocks_fp32(
            static_cast<const float *>(device_zero.get()),
            static_cast<const float *>(device_zero.get()), space, 1,
            material, state != 0,
            static_cast<std::uint32_t *>(device_count.get()),
            static_cast<std::uint32_t *>(device_validation.get()));
        meep_cuda::synchronize();
        const std::uint32_t device_active =
            copy_scalar_from_device<std::uint32_t>(device_count.get());
        const std::uint32_t device_flag =
            copy_scalar_from_device<std::uint32_t>(
                device_validation.get());
        require(device_active == (host_safe ? 0U : 1U) &&
                    ((device_flag &
                      meep_cuda::lorentzian_sparse_coefficient_unsafe_fp32) !=
                     0) == !host_safe,
                "host/device sparse coefficient predicate disagrees");
      }
}

void run_state_form(bool increment_state) {
  const meep_cuda::index_space_fp32 space = irregular_space();
  const std::size_t point_count =
      space.extent1 * space.extent2 * space.extent3;
  const std::size_t array_count = 1500;
  const std::vector<std::ptrdiff_t> owned =
      decoded_indices(space, point_count);
  require(std::is_sorted(owned.begin(), owned.end()),
          "test index space must decode in sorted order");
  require(static_cast<std::size_t>(owned.back()) < array_count,
          "test index space exceeds allocation");

  std::vector<float> initial_p(array_count, 0.0f);
  std::vector<float> initial_aux(array_count, 0.0f);
  std::vector<float> sigma(array_count, 0.0f);
  std::vector<float> finite_field(array_count, 0.0f);
  for (std::size_t index = 0; index < array_count; ++index)
    finite_field[index] =
        static_cast<float>(std::sin(0.013 * static_cast<double>(index + 1)));

  for (std::size_t point = 0; point < point_count; ++point) {
    const std::size_t index = static_cast<std::size_t>(owned[point]);
    if (point % 97 == 0) initial_p[index] = 0.125f;
    if (point % 131 == 0) initial_aux[index] = -0.0f;
    if (point % 173 == 0)
      sigma[index] = std::numeric_limits<float>::quiet_NaN();
    else if (point % 41 == 0)
      sigma[index] = 0.21f;
  }

  device_pointer device_candidate_p = copy_to_device(initial_p);
  device_pointer device_candidate_aux = copy_to_device(initial_aux);
  device_pointer device_baseline_p = copy_to_device(initial_p);
  device_pointer device_baseline_aux = copy_to_device(initial_aux);
  device_pointer device_sigma = copy_to_device(sigma);
  device_pointer device_field = copy_to_device(finite_field);

  meep_cuda::lorentzian_material_fp32 material{};
  material.sigma = static_cast<const float *>(device_sigma.get());
  material.gamma_inverse = 0.91f;
  material.gamma_previous = 0.87f;
  material.omega_dt_squared = 0.12f;
  material.omega_dt_squared_denominator = 0.12f;

  const std::size_t logical_blocks =
      meep_cuda::lorentzian_sparse_compaction_block_count(point_count);
  require(logical_blocks == 5,
          "fixed compaction block count is incorrect");
  std::vector<std::uint32_t> zero_counts(logical_blocks, 0);
  device_pointer device_counts = copy_to_device(zero_counts);
  const std::uint32_t zero = 0;
  device_pointer device_validation = copy_scalar_to_device(zero);
  meep_cuda::count_lorentzian_sparse_active_blocks_fp32(
      static_cast<const float *>(device_candidate_p.get()),
      static_cast<const float *>(device_candidate_aux.get()), space,
      point_count, material, increment_state,
      static_cast<std::uint32_t *>(device_counts.get()),
      static_cast<std::uint32_t *>(device_validation.get()));
  const std::vector<std::uint32_t> counts =
      copy_from_device<std::uint32_t>(device_counts.get(), logical_blocks);

  std::vector<std::ptrdiff_t> expected_active;
  std::vector<std::uint32_t> expected_counts(logical_blocks, 0);
  for (std::size_t point = 0; point < point_count; ++point) {
    const std::size_t index = static_cast<std::size_t>(owned[point]);
    if (meep_cuda::detail::lorentzian_sparse_point_active_fp32(
            initial_p[index], initial_aux[index], sigma[index])) {
      expected_active.push_back(owned[point]);
      ++expected_counts[point /
                        meep_cuda::lorentzian_sparse_compaction_block_size];
    }
  }
  require(counts == expected_counts,
          "device sparse block counts differ from exact host predicate");
  require(copy_scalar_from_device<std::uint32_t>(device_validation.get()) == 0,
          "safe device coefficient tuple raised validation flag");
  require(!expected_active.empty() && expected_active.size() < point_count,
          "mixed sparse test did not produce a mixed active set");

  std::vector<std::uint64_t> offsets(logical_blocks, 0);
  std::uint64_t active_count = 0;
  for (std::size_t block = 0; block < logical_blocks; ++block) {
    offsets[block] = active_count;
    active_count += counts[block];
  }
  require(active_count == expected_active.size(),
          "checked active prefix total differs from oracle");
  device_pointer device_offsets = copy_to_device(offsets);
  std::vector<std::ptrdiff_t> empty_active(expected_active.size(), -1);
  device_pointer device_active = copy_to_device(empty_active);
  meep_cuda::fill_lorentzian_sparse_active_indices_fp32(
      static_cast<const float *>(device_candidate_p.get()),
      static_cast<const float *>(device_candidate_aux.get()), space,
      point_count, material, increment_state,
      static_cast<const std::uint64_t *>(device_offsets.get()),
      expected_active.size(),
      static_cast<std::ptrdiff_t *>(device_active.get()),
      static_cast<std::uint32_t *>(device_validation.get()));
  meep_cuda::synchronize();
  require(copy_scalar_from_device<std::uint32_t>(device_validation.get()) == 0,
          "valid sparse fill raised validation flag");
  require(copy_from_device<std::ptrdiff_t>(
              device_active.get(), expected_active.size()) == expected_active,
          "sparse fill is not exact and deterministically sorted");

  const auto require_invalid_offsets =
      [&](const std::vector<std::uint64_t> &invalid_offsets,
          std::size_t declared_active_count, const char *label) {
    require(invalid_offsets.size() == offsets.size(),
            "invalid-offset fixture size mismatch");
    meep_cuda::copy_to_device(
        device_offsets.get(), invalid_offsets.data(),
        invalid_offsets.size() * sizeof(std::uint64_t));
    meep_cuda::copy_to_device(
        device_validation.get(), &zero, sizeof(zero));
    meep_cuda::fill_lorentzian_sparse_active_indices_fp32(
        static_cast<const float *>(device_candidate_p.get()),
        static_cast<const float *>(device_candidate_aux.get()), space,
        point_count, material, increment_state,
        static_cast<const std::uint64_t *>(device_offsets.get()),
        declared_active_count,
        static_cast<std::ptrdiff_t *>(device_active.get()),
        static_cast<std::uint32_t *>(device_validation.get()));
    meep_cuda::synchronize();
    const std::uint32_t validation =
        copy_scalar_from_device<std::uint32_t>(device_validation.get());
    require((validation & meep_cuda::lorentzian_sparse_fill_invalid_fp32) != 0,
            label);
  };

  require(expected_active.size() > 1 && logical_blocks > 1,
          "invalid-offset fixtures require multiple active points and blocks");
  require_invalid_offsets(
      offsets, expected_active.size() - 1,
      "truncated final count did not fail closed");
  std::vector<std::uint64_t> duplicate_offsets = offsets;
  duplicate_offsets[1] = duplicate_offsets[0];
  require_invalid_offsets(
      duplicate_offsets, expected_active.size(),
      "duplicate block offset did not fail closed");
  std::vector<std::uint64_t> gapped_offsets = offsets;
  ++gapped_offsets[1];
  require_invalid_offsets(
      gapped_offsets, expected_active.size(),
      "gapped block offset did not fail closed");
  std::vector<std::uint64_t> overflowing_offsets = offsets;
  overflowing_offsets[0] = std::numeric_limits<std::uint64_t>::max();
  require_invalid_offsets(
      overflowing_offsets, expected_active.size(),
      "near-UINT64_MAX block offset did not fail closed");
  require(expected_active.size() + 1 <= point_count,
          "final-count mismatch fixture exceeds structured count");
  require_invalid_offsets(
      offsets, expected_active.size() + 1,
      "oversized final count did not fail closed");

  // Rebuild private scratch after the deliberately malformed fill attempts.
  meep_cuda::copy_to_device(
      device_offsets.get(), offsets.data(),
      offsets.size() * sizeof(std::uint64_t));
  meep_cuda::copy_to_device(device_validation.get(), &zero, sizeof(zero));
  meep_cuda::fill_lorentzian_sparse_active_indices_fp32(
      static_cast<const float *>(device_candidate_p.get()),
      static_cast<const float *>(device_candidate_aux.get()), space,
      point_count, material, increment_state,
      static_cast<const std::uint64_t *>(device_offsets.get()),
      expected_active.size(),
      static_cast<std::ptrdiff_t *>(device_active.get()),
      static_cast<std::uint32_t *>(device_validation.get()));
  meep_cuda::synchronize();
  require(copy_scalar_from_device<std::uint32_t>(device_validation.get()) == 0,
          "valid sparse fill did not recover after rejected scratch builds");
  require(copy_from_device<std::ptrdiff_t>(
              device_active.get(), expected_active.size()) == expected_active,
          "recovered sparse fill differs from exact structured order");

  meep_cuda::lorentzian_sparse_execution_ledger_fp32 ledger{};
  device_pointer device_ledger = copy_scalar_to_device(ledger);
  device_pointer device_sticky = copy_scalar_to_device(zero);
  meep_cuda::scan_lorentzian_sparse_field_finite_fp32(
      static_cast<const float *>(device_field.get()), space, point_count,
      static_cast<std::uint32_t *>(device_sticky.get()),
      static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
          device_ledger.get()));
  if (increment_state) {
    meep_cuda::update_lorentzian_increment_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
    meep_cuda::update_lorentzian_increment_structured_fp32(
        static_cast<float *>(device_baseline_p.get()),
        static_cast<float *>(device_baseline_aux.get()),
        static_cast<const float *>(device_field.get()), nullptr, nullptr,
        space, point_count, 0, 0, 0, material);
  }
  else {
    meep_cuda::update_lorentzian_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
    meep_cuda::update_lorentzian_structured_fp32(
        static_cast<float *>(device_baseline_p.get()),
        static_cast<float *>(device_baseline_aux.get()),
        static_cast<const float *>(device_field.get()), nullptr, nullptr,
        space, point_count, 0, 0, 0, material);
  }
  meep_cuda::synchronize();
  require_bitwise_equal(
      copy_from_device<float>(device_candidate_p.get(), array_count),
      copy_from_device<float>(device_baseline_p.get(), array_count),
      "finite guarded sparse polarization");
  require_bitwise_equal(
      copy_from_device<float>(device_candidate_aux.get(), array_count),
      copy_from_device<float>(device_baseline_aux.get(), array_count),
      "finite guarded sparse auxiliary");

  ledger = copy_scalar_from_device<
      meep_cuda::lorentzian_sparse_execution_ledger_fp32>(
      device_ledger.get());
  require(ledger.guard_scan_calls == 1 &&
              ledger.guard_scan_points == point_count &&
              ledger.sparse_update_calls == 1 &&
              ledger.sparse_update_points == expected_active.size() &&
              ledger.dense_fallback_calls == 0 &&
              ledger.counter_overflow == 0,
          "finite sparse causal ledger is incorrect");

  std::vector<float> nonfinite_field = finite_field;
  const std::size_t poisoned_index = static_cast<std::size_t>(owned[1]);
  nonfinite_field[poisoned_index] =
      std::numeric_limits<float>::infinity();
  const std::size_t nan_index = static_cast<std::size_t>(owned[2]);
  nonfinite_field[nan_index] =
      std::numeric_limits<float>::quiet_NaN();
  meep_cuda::copy_to_device(
      device_candidate_p.get(), initial_p.data(),
      initial_p.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_candidate_aux.get(), initial_aux.data(),
      initial_aux.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_baseline_p.get(), initial_p.data(),
      initial_p.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_baseline_aux.get(), initial_aux.data(),
      initial_aux.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_field.get(), nonfinite_field.data(),
      nonfinite_field.size() * sizeof(float));

  meep_cuda::scan_lorentzian_sparse_field_finite_fp32(
      static_cast<const float *>(device_field.get()), space, point_count,
      static_cast<std::uint32_t *>(device_sticky.get()),
      static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
          device_ledger.get()));
  if (increment_state) {
    meep_cuda::update_lorentzian_increment_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
    meep_cuda::update_lorentzian_increment_structured_fp32(
        static_cast<float *>(device_baseline_p.get()),
        static_cast<float *>(device_baseline_aux.get()),
        static_cast<const float *>(device_field.get()), nullptr, nullptr,
        space, point_count, 0, 0, 0, material);
  }
  else {
    meep_cuda::update_lorentzian_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
    meep_cuda::update_lorentzian_structured_fp32(
        static_cast<float *>(device_baseline_p.get()),
        static_cast<float *>(device_baseline_aux.get()),
        static_cast<const float *>(device_field.get()), nullptr, nullptr,
        space, point_count, 0, 0, 0, material);
  }
  meep_cuda::synchronize();
  require(copy_scalar_from_device<std::uint32_t>(device_sticky.get()) == 1,
          "non-finite field did not poison sticky guard");
  require_bitwise_equal(
      copy_from_device<float>(device_candidate_p.get(), array_count),
      copy_from_device<float>(device_baseline_p.get(), array_count),
      "poisoned dense-fallback polarization");
  require_bitwise_equal(
      copy_from_device<float>(device_candidate_aux.get(), array_count),
      copy_from_device<float>(device_baseline_aux.get(), array_count),
      "poisoned dense-fallback auxiliary");

  // A finite field never clears the sticky flag; the next guarded update must
  // still execute the complete dense recurrence.
  meep_cuda::copy_to_device(
      device_candidate_p.get(), initial_p.data(),
      initial_p.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_candidate_aux.get(), initial_aux.data(),
      initial_aux.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_baseline_p.get(), initial_p.data(),
      initial_p.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_baseline_aux.get(), initial_aux.data(),
      initial_aux.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_field.get(), finite_field.data(),
      finite_field.size() * sizeof(float));
  meep_cuda::scan_lorentzian_sparse_field_finite_fp32(
      static_cast<const float *>(device_field.get()), space, point_count,
      static_cast<std::uint32_t *>(device_sticky.get()),
      static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
          device_ledger.get()));
  if (increment_state) {
    meep_cuda::update_lorentzian_increment_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
    meep_cuda::update_lorentzian_increment_structured_fp32(
        static_cast<float *>(device_baseline_p.get()),
        static_cast<float *>(device_baseline_aux.get()),
        static_cast<const float *>(device_field.get()), nullptr, nullptr,
        space, point_count, 0, 0, 0, material);
  }
  else {
    meep_cuda::update_lorentzian_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
    meep_cuda::update_lorentzian_structured_fp32(
        static_cast<float *>(device_baseline_p.get()),
        static_cast<float *>(device_baseline_aux.get()),
        static_cast<const float *>(device_field.get()), nullptr, nullptr,
        space, point_count, 0, 0, 0, material);
  }
  meep_cuda::synchronize();
  require_bitwise_equal(
      copy_from_device<float>(device_candidate_p.get(), array_count),
      copy_from_device<float>(device_baseline_p.get(), array_count),
      "sticky dense-fallback polarization");
  ledger = copy_scalar_from_device<
      meep_cuda::lorentzian_sparse_execution_ledger_fp32>(
      device_ledger.get());
  require(ledger.guard_scan_calls == 3 &&
              ledger.guard_scan_points == 3 * point_count &&
              ledger.sparse_update_calls == 1 &&
              ledger.sparse_update_points == expected_active.size() &&
              ledger.dense_fallback_calls == 2 &&
              ledger.dense_fallback_points == 2 * point_count,
          "sticky dense-fallback causal ledger is incorrect");

  // A rejected compaction plan is independently fail-closed even when the
  // field-finiteness guard is clean.  The guarded update must ignore the
  // private sparse scratch and execute the complete structured recurrence.
  meep_cuda::copy_to_device(
      device_candidate_p.get(), initial_p.data(),
      initial_p.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_candidate_aux.get(), initial_aux.data(),
      initial_aux.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_baseline_p.get(), initial_p.data(),
      initial_p.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_baseline_aux.get(), initial_aux.data(),
      initial_aux.size() * sizeof(float));
  meep_cuda::copy_to_device(
      device_field.get(), finite_field.data(),
      finite_field.size() * sizeof(float));
  const std::uint32_t rejected_plan =
      meep_cuda::lorentzian_sparse_fill_invalid_fp32;
  ledger = {};
  meep_cuda::copy_to_device(
      device_validation.get(), &rejected_plan, sizeof(rejected_plan));
  meep_cuda::copy_to_device(device_sticky.get(), &zero, sizeof(zero));
  meep_cuda::copy_to_device(device_ledger.get(), &ledger, sizeof(ledger));
  if (increment_state) {
    meep_cuda::update_lorentzian_increment_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
    meep_cuda::update_lorentzian_increment_structured_fp32(
        static_cast<float *>(device_baseline_p.get()),
        static_cast<float *>(device_baseline_aux.get()),
        static_cast<const float *>(device_field.get()), nullptr, nullptr,
        space, point_count, 0, 0, 0, material);
  }
  else {
    meep_cuda::update_lorentzian_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
    meep_cuda::update_lorentzian_structured_fp32(
        static_cast<float *>(device_baseline_p.get()),
        static_cast<float *>(device_baseline_aux.get()),
        static_cast<const float *>(device_field.get()), nullptr, nullptr,
        space, point_count, 0, 0, 0, material);
  }
  meep_cuda::synchronize();
  require_bitwise_equal(
      copy_from_device<float>(device_candidate_p.get(), array_count),
      copy_from_device<float>(device_baseline_p.get(), array_count),
      "rejected-plan dense-fallback polarization");
  require_bitwise_equal(
      copy_from_device<float>(device_candidate_aux.get(), array_count),
      copy_from_device<float>(device_baseline_aux.get(), array_count),
      "rejected-plan dense-fallback auxiliary");
  ledger = copy_scalar_from_device<
      meep_cuda::lorentzian_sparse_execution_ledger_fp32>(
      device_ledger.get());
  require(ledger.guard_scan_calls == 0 &&
              ledger.sparse_update_calls == 0 &&
              ledger.dense_fallback_calls == 1 &&
              ledger.dense_fallback_points == point_count &&
              ledger.counter_overflow == 0,
          "rejected-plan dense-fallback causal ledger is incorrect");

  // The device-side coefficient predicate must fail closed independently of
  // the host predicate and classify the complete domain as active.
  meep_cuda::lorentzian_material_fp32 unsafe_material = material;
  unsafe_material.gamma_inverse = -1.0f;
  unsafe_material.gamma_previous = 1.0f;
  unsafe_material.omega_dt_squared = 0.0f;
  unsafe_material.omega_dt_squared_denominator = 0.0f;
  meep_cuda::copy_to_device(
      device_counts.get(), zero_counts.data(),
      zero_counts.size() * sizeof(std::uint32_t));
  meep_cuda::copy_to_device(device_validation.get(), &zero, sizeof(zero));
  meep_cuda::count_lorentzian_sparse_active_blocks_fp32(
      static_cast<const float *>(device_candidate_p.get()),
      static_cast<const float *>(device_candidate_aux.get()), space,
      point_count, unsafe_material, increment_state,
      static_cast<std::uint32_t *>(device_counts.get()),
      static_cast<std::uint32_t *>(device_validation.get()));
  const std::vector<std::uint32_t> unsafe_counts =
      copy_from_device<std::uint32_t>(device_counts.get(), logical_blocks);
  require(std::accumulate(unsafe_counts.begin(), unsafe_counts.end(),
                          std::uint64_t{0}) == point_count,
          "unsafe device coefficient tuple did not force all-active");
  require((copy_scalar_from_device<std::uint32_t>(device_validation.get()) &
           meep_cuda::lorentzian_sparse_coefficient_unsafe_fp32) != 0,
          "unsafe device coefficient tuple did not set plan flag");

  // All-positive-zero state is an exact empty plan and requires no index
  // allocation. The fill operation accepts a null active-index pointer.
  std::vector<float> all_zero(array_count, 0.0f);
  device_pointer device_zero = copy_to_device(all_zero);
  meep_cuda::lorentzian_material_fp32 zero_material = material;
  zero_material.sigma = static_cast<const float *>(device_zero.get());
  meep_cuda::copy_to_device(device_validation.get(), &zero, sizeof(zero));
  meep_cuda::count_lorentzian_sparse_active_blocks_fp32(
      static_cast<const float *>(device_zero.get()),
      static_cast<const float *>(device_zero.get()), space, point_count,
      zero_material, increment_state,
      static_cast<std::uint32_t *>(device_counts.get()),
      static_cast<std::uint32_t *>(device_validation.get()));
  const std::vector<std::uint32_t> empty_counts =
      copy_from_device<std::uint32_t>(device_counts.get(), logical_blocks);
  require(std::accumulate(empty_counts.begin(), empty_counts.end(),
                          std::uint64_t{0}) == 0,
          "positive-zero device state was not an empty plan");
  std::fill(offsets.begin(), offsets.end(), 0);
  meep_cuda::copy_to_device(
      device_offsets.get(), offsets.data(),
      offsets.size() * sizeof(std::uint64_t));
  meep_cuda::copy_to_device(device_validation.get(), &zero, sizeof(zero));
  meep_cuda::fill_lorentzian_sparse_active_indices_fp32(
      static_cast<const float *>(device_zero.get()),
      static_cast<const float *>(device_zero.get()), space, point_count,
      zero_material, increment_state,
      static_cast<const std::uint64_t *>(device_offsets.get()), 0, nullptr,
      static_cast<std::uint32_t *>(device_validation.get()));
  meep_cuda::synchronize();
  require(copy_scalar_from_device<std::uint32_t>(device_validation.get()) == 0,
          "empty sparse fill raised validation flag");

  // Device ledgers saturate rather than wrapping and make overflow visible.
  ledger = {};
  ledger.guard_scan_calls = std::numeric_limits<std::uint64_t>::max();
  ledger.guard_scan_points = std::numeric_limits<std::uint64_t>::max();
  ledger.sparse_update_calls = std::numeric_limits<std::uint64_t>::max();
  ledger.sparse_update_points = std::numeric_limits<std::uint64_t>::max();
  ledger.dense_fallback_calls = std::numeric_limits<std::uint64_t>::max();
  ledger.dense_fallback_points = std::numeric_limits<std::uint64_t>::max();
  meep_cuda::copy_to_device(device_ledger.get(), &ledger, sizeof(ledger));
  meep_cuda::copy_to_device(device_sticky.get(), &zero, sizeof(zero));
  meep_cuda::scan_lorentzian_sparse_field_finite_fp32(
      static_cast<const float *>(device_field.get()), space, point_count,
      static_cast<std::uint32_t *>(device_sticky.get()),
      static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
          device_ledger.get()));
  if (increment_state)
    meep_cuda::update_lorentzian_increment_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
  else
    meep_cuda::update_lorentzian_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
  const std::uint32_t one = 1;
  meep_cuda::copy_to_device(device_sticky.get(), &one, sizeof(one));
  if (increment_state)
    meep_cuda::update_lorentzian_increment_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
  else
    meep_cuda::update_lorentzian_sparse_guarded_fp32(
        static_cast<float *>(device_candidate_p.get()),
        static_cast<float *>(device_candidate_aux.get()),
        static_cast<const float *>(device_field.get()),
        static_cast<const std::ptrdiff_t *>(device_active.get()),
        expected_active.size(), space, point_count, material,
        static_cast<const std::uint32_t *>(device_validation.get()),
        static_cast<const std::uint32_t *>(device_sticky.get()),
        static_cast<meep_cuda::lorentzian_sparse_execution_ledger_fp32 *>(
            device_ledger.get()));
  meep_cuda::synchronize();
  ledger = copy_scalar_from_device<
      meep_cuda::lorentzian_sparse_execution_ledger_fp32>(
      device_ledger.get());
  require(ledger.guard_scan_calls ==
                  std::numeric_limits<std::uint64_t>::max() &&
              ledger.guard_scan_points ==
                  std::numeric_limits<std::uint64_t>::max() &&
              ledger.sparse_update_calls ==
                  std::numeric_limits<std::uint64_t>::max() &&
              ledger.sparse_update_points ==
                  std::numeric_limits<std::uint64_t>::max() &&
              ledger.dense_fallback_calls ==
                  std::numeric_limits<std::uint64_t>::max() &&
              ledger.dense_fallback_points ==
                  std::numeric_limits<std::uint64_t>::max() &&
              ledger.counter_overflow == 1,
          "sparse device ledger did not report saturating overflow");
}

} // namespace

int main() {
  try {
    const int device = selected_test_device();
    if (device < 0) {
      std::cout << "SKIP: no compatible CUDA device\n";
      return 77;
    }
    meep_cuda::select_device(device);
    check_compaction_boundaries_and_2d_ordering();
    check_device_coefficient_matrix();
    run_state_form(false);
    run_state_form(true);
    std::cout << "PASS: deterministic sparse Lorentzian compaction, guarded "
                 "previous/increment updates, sticky fallback, and causal "
                 "ledger\n";
    return 0;
  }
  catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
