#include "meep/gpu.hpp"
#include "meep/meep-config.h"
#include "gpu_backend_internal.hpp"
#include "gpu_grid_index.hpp"
#include "meep_cuda/detail/index_space.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(meep::gpu::resident_statistics) ==
                  4 * sizeof(std::uint64_t),
              "resident_statistics ABI must remain four uint64_t fields");
static_assert(sizeof(meep::gpu::phase_batch_policy_statistics) ==
                  12 * sizeof(std::uint64_t),
              "phase_batch_policy_statistics ABI must remain twelve "
              "uint64_t fields");

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

void require_cuda_active_hot_view(bool expected, const char *message) {
  for (int iteration = 0; iteration < 1024; ++iteration)
    require(meep::gpu::detail::cuda_active() == expected, message);
}

void check_structured_grid_indices() {
  const meep::grid_volume gv = meep::vol3d(2.0, 3.0, 4.0, 2.0);
  const meep::component component = meep::Ex;
  const meep::ivec is = gv.little_owned_corner0(component);
  const meep::ivec ie = gv.big_corner();
  const meep::gpu::detail::index_space_fp32 host_space =
      meep::gpu::detail::make_index_space_fp32(
          gv, is, ie, meep::X, meep::Y);
  const meep_cuda::index_space_fp32 device_space = {
      host_space.field_start,
      host_space.extent1,
      host_space.extent2,
      host_space.extent3,
      host_space.field_stride1,
      host_space.field_stride2,
      host_space.field_stride3,
      host_space.coefficient_start,
      host_space.coefficient_stride1,
      host_space.coefficient_stride2,
      host_space.coefficient_stride3,
      host_space.coefficient2_start,
      host_space.coefficient2_stride1,
      host_space.coefficient2_stride2,
      host_space.coefficient2_stride3};

  std::size_t linear = 0;
  LOOP_OVER_IVECS(gv, is, ie, field_index) {
    const meep_cuda::detail::decoded_index_fp32 decoded =
        meep_cuda::detail::decode_index_space_fp32(device_space, linear);
    require(decoded.field == field_index,
            "structured field index disagrees with LOOP_OVER_IVECS");
    const int expected_x =
        is.in_direction(meep::X) -
        gv.little_corner().in_direction(meep::X) +
        (gv.yucky_direction(0) == meep::X ? 2 * loop_i1 : 0) +
        (gv.yucky_direction(1) == meep::X ? 2 * loop_i2 : 0) +
        (gv.yucky_direction(2) == meep::X ? 2 * loop_i3 : 0);
    const int expected_y =
        is.in_direction(meep::Y) -
        gv.little_corner().in_direction(meep::Y) +
        (gv.yucky_direction(0) == meep::Y ? 2 * loop_i1 : 0) +
        (gv.yucky_direction(1) == meep::Y ? 2 * loop_i2 : 0) +
        (gv.yucky_direction(2) == meep::Y ? 2 * loop_i3 : 0);
    require(decoded.coefficient == expected_x,
            "structured first PML index disagrees with LOOP_OVER_IVECS");
    require(decoded.coefficient2 == expected_y,
            "structured second PML index disagrees with LOOP_OVER_IVECS");
    ++linear;
  }
  require(linear ==
              host_space.extent1 * host_space.extent2 *
                  host_space.extent3,
          "structured index point count disagrees with LOOP_OVER_IVECS");
}

void check_rank_device_mapping() {
  const int devices[] = {2, 4, 7, 9};
  require(meep::gpu::detail::choose_rank_device_ordinal(devices, 4, 0, false) == 2,
          "node rank zero must select the first compatible GPU");
  require(meep::gpu::detail::choose_rank_device_ordinal(devices, 4, 3, false) == 9,
          "node rank must select the matching compatible GPU");
  require(meep::gpu::detail::choose_rank_device_ordinal(devices, 4, 5, true) == 4,
          "explicit oversubscription must wrap node ranks");
  const int isolated[] = {0};
  require(meep::gpu::detail::choose_rank_device_ordinal(isolated, 1, 19, false) == 0,
          "one-device-per-rank visibility must always select ordinal zero");
  require(meep::gpu::detail::choose_rank_device_ordinal(nullptr, 0, 0, false) == -1,
          "an empty compatible-device set must return no device");

  bool rejected_oversubscription = false;
  try {
    (void)meep::gpu::detail::choose_rank_device_ordinal(devices, 4, 4, false);
  }
  catch (const std::runtime_error &) {
    rejected_oversubscription = true;
  }
  require(rejected_oversubscription,
          "implicit multi-rank GPU oversubscription must be rejected");
}

void expect_tile_rejection(const meep::grid_volume &root,
                           const std::vector<meep::grid_volume> &tiles, std::size_t base,
                           const char *message) {
  bool rejected = false;
  try {
    meep::check_tiles(root, tiles, base);
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, message);
}

void check_linear_tile_validation() {
  int legacy_split_point = -1;
  meep::direction legacy_split_direction = meep::X;
  meep::vol1d(2.0, 1.0).tile_split(legacy_split_point, legacy_split_direction);
  require(legacy_split_point == 0 && legacy_split_direction == meep::NO_DIRECTION,
          "public tiny-volume tile_split behavior must remain backward compatible");
  meep::volcyl(4.0, 2.0, 1.0).tile_split(legacy_split_point, legacy_split_direction);
  require(legacy_split_point == 2 && legacy_split_direction == meep::X,
          "public cylindrical tile_split priority must remain backward compatible");

  const std::vector<meep::grid_volume> roots = {
      meep::vol1d(17.0, 1.0), meep::vol2d(15.0, 13.0, 1.0),
      meep::vol3d(11.0, 9.0, 7.0, 1.0), meep::volcyl(13.0, 11.0, 1.0)};
  const std::size_t bases[] = {2, 7, 32, 128};
  for (auto root : roots) {
    root.shift_origin(meep::start_at_direction(root.dim), 18);
    for (const std::size_t base : bases) {
      std::vector<meep::grid_volume> tiles;
      meep::split_into_tiles(root, &tiles, base);
      meep::check_tiles(root, tiles, base);
      meep::check_tiles(root, tiles); // independent order-agnostic small-volume oracle
    }
  }

  const meep::grid_volume root = meep::vol3d(11.0, 9.0, 7.0, 1.0);
  std::vector<meep::grid_volume> tiles;
  meep::split_into_tiles(root, &tiles, 32);

  std::vector<meep::grid_volume> missing = tiles;
  missing.pop_back();
  expect_tile_rejection(root, missing, 32, "linear tile validation must reject a missing leaf");

  std::vector<meep::grid_volume> trailing = tiles;
  trailing.push_back(tiles.back());
  expect_tile_rejection(root, trailing, 32,
                        "linear tile validation must reject a trailing duplicate leaf");

  std::vector<meep::grid_volume> reordered = tiles;
  std::swap(reordered[0], reordered[1]);
  expect_tile_rejection(root, reordered, 32,
                        "generated-partition validation must reject reordered leaves");

  std::vector<meep::grid_volume> shifted = tiles;
  shifted[0].shift_origin(meep::X, 2);
  expect_tile_rejection(root, shifted, 32,
                        "linear tile validation must reject shifted leaf geometry");

  std::vector<meep::grid_volume> resized = tiles;
  resized[0].set_num_direction(meep::X, resized[0].nx() + 1);
  expect_tile_rejection(root, resized, 32,
                        "linear tile validation must reject resized leaf geometry");

  std::vector<meep::grid_volume> wrong_resolution = tiles;
  wrong_resolution[0].a *= 2.0;
  expect_tile_rejection(root, wrong_resolution, 32,
                        "linear tile validation must reject inconsistent resolution");

  std::vector<meep::grid_volume> wrong_dimension = tiles;
  wrong_dimension[0].dim = static_cast<meep::ndim>(99);
  expect_tile_rejection(root, wrong_dimension, 32,
                        "linear tile validation must reject invalid dimensions safely");
  expect_tile_rejection(root, tiles, 128,
                        "linear tile validation must reject a partition generated for another base");

  bool base_one_rejected = false;
  try {
    std::vector<meep::grid_volume> invalid_base_tiles;
    meep::split_into_tiles(root, &invalid_base_tiles, 1);
  }
  catch (const std::runtime_error &) {
    base_one_rejected = true;
  }
  require(base_one_rejected, "tile base one must be rejected instead of selecting NO_DIRECTION");

  meep::grid_volume overflow_root = meep::vol3d(1.0, 1.0, 1.0, 1.0);
  overflow_root.set_num_direction(meep::X, std::numeric_limits<int>::max());
  overflow_root.set_num_direction(meep::Y, std::numeric_limits<int>::max());
  overflow_root.set_num_direction(meep::Z, std::numeric_limits<int>::max());
  expect_tile_rejection(overflow_root, {overflow_root}, 2,
                        "tile validation must reject a grid-point count overflow safely");

  bool unsplit_end_overflow_rejected = false;
  try {
    meep::grid_volume unsplit_end_overflow = meep::vol1d(1.0, 1.0);
    unsplit_end_overflow.set_origin(meep::Z, std::numeric_limits<int>::max() - 1);
    std::vector<meep::grid_volume> unsplit_end_overflow_tiles;
    meep::split_into_tiles(unsplit_end_overflow, &unsplit_end_overflow_tiles, 2);
  }
  catch (const std::runtime_error &) {
    unsplit_end_overflow_rejected = true;
  }
  require(unsplit_end_overflow_rejected,
          "an unsplit tile with an overflowing end corner must be rejected safely");

  bool origin_overflow_rejected = false;
  try {
    meep::grid_volume origin_overflow = meep::vol1d(4.0, 1.0);
    origin_overflow.set_origin(meep::Z, std::numeric_limits<int>::max() - 2);
    std::vector<meep::grid_volume> origin_overflow_tiles;
    meep::split_into_tiles(origin_overflow, &origin_overflow_tiles, 2);
  }
  catch (const std::runtime_error &) {
    origin_overflow_rejected = true;
  }
  require(origin_overflow_rejected, "tile splitting must reject an integer origin overflow safely");

  meep::grid_volume stale_origin = meep::vol1d(1.0, 1.0);
  stale_origin.set_origin(meep::Z, 2);
  stale_origin.a = 2.0;
  stale_origin.inva = 0.5;
  expect_tile_rejection(stale_origin, {stale_origin}, 2,
                        "tile validation must reject a stale cached physical origin");

  const meep::grid_volume large_root = meep::vol3d(384.0, 384.0, 192.0, 1.0);
  std::vector<meep::grid_volume> large_tiles;
  meep::split_into_tiles(large_root, &large_tiles, 128);
  require(large_tiles.size() == 294912,
          "384x384x192/base128 partition has an unexpected leaf count");
  meep::check_tiles(large_root, large_tiles, 128);
}

void check_fields_tile_base_lifecycle() {
  const meep::grid_volume gv = meep::vol2d(1.0, 1.0, 4.0);
  meep::structure_chunk structure_chunk(gv, gv.surroundings(), 0.5, 0);
  const int initial_refcount = structure_chunk.refcount;
  for (const int invalid_base : {-1, 1}) {
    bool rejected = false;
    try {
      meep::fields_chunk invalid(&structure_chunk, "", 0.0, 0.0, true, 0, invalid_base,
                                 {0.0, 0.0, 0.0});
    }
    catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "direct fields_chunk construction must reject an invalid tile base");
    require(structure_chunk.refcount == initial_refcount,
            "rejected fields_chunk construction must not leak a structure refcount");
  }
}

void check_sourcedata_manifest() {
  meep::sourcedata data;
  data.near_fd_comp = meep::Ez;
  data.idx_arr = {3, 7, 11};
  data.fc_idx = 2;
  data.amp_arr = {{1.0, -2.0}, {3.5, 4.25}};
  const std::string baseline = data.gpmeep_manifest();
  require(baseline.find("\"schema\":\"gpmeep-sourcedata-v1\"") !=
              std::string::npos,
          "sourcedata manifest must expose its schema");
  require(baseline.find("\"index_count\":3") != std::string::npos &&
              baseline.find("\"amplitude_count\":2") != std::string::npos,
          "sourcedata manifest must expose exact vector lengths");
  require(meep::sourcedata(data).gpmeep_manifest() == baseline,
          "identical sourcedata values must produce a stable manifest");

  meep::sourcedata changed = data;
  changed.idx_arr[1] += 1;
  require(changed.gpmeep_manifest() != baseline,
          "sourcedata index mutation must change the manifest");
  changed = data;
  changed.amp_arr[0] += std::complex<double>(0.0, 0.25);
  require(changed.gpmeep_manifest() != baseline,
          "sourcedata amplitude mutation must change the manifest");
  changed = data;
  changed.fc_idx += 1;
  require(changed.gpmeep_manifest() != baseline,
          "sourcedata chunk mutation must change the manifest");
  changed = data;
  changed.near_fd_comp = meep::Ex;
  require(changed.gpmeep_manifest() != baseline,
          "sourcedata component mutation must change the manifest");
}

double unit_permittivity(const meep::vec &) { return 1.0; }

void check_native_source_cycle_manifest_rejection() {
  const meep::grid_volume gv = meep::vol2d(1.0, 1.0, 4.0);
  meep::structure structure(gv, unit_permittivity, meep::no_pml());
  meep::fields fields(&structure);
  meep::src_time *source = new meep::src_time();
  fields.sources = source;
  source->next = source;
  bool rejected = false;
  try {
    (void)fields.gpmeep_state_manifest();
  }
  catch (const std::runtime_error &) {
    rejected = true;
  }
  // Restore unique ownership before either object is destroyed.
  source->next = nullptr;
  fields.sources = nullptr;
  delete source;
  require(rejected,
          "native fields manifest must reject an owning source-list cycle");
}

} // namespace

int main(int argc, char **argv) {
  // Keep the pure host validation ahead of MPI initialization.  Several
  // cases deliberately feed malformed tile partitions to APIs whose
  // single-process contract is to throw.  Once MPI is initialized those
  // same diagnostics correctly become collective MPI_Abort failures, which
  // would terminate Automake's ordinary two-rank test runner before the
  // rejection assertions can observe them.
  check_structured_grid_indices();
  check_sourcedata_manifest();
  check_rank_device_mapping();
  check_linear_tile_validation();
  check_fields_tile_base_lifecycle();

  // The native fields/source ownership regression needs an initialized Meep
  // MPI lifetime, including when the test is launched through RUNCODE.
  meep::initialize mpi(argc, argv);
  check_native_source_cycle_manifest_rejection();
  require(meep::gpu::backend_compiled() == (MEEP_HAVE_CUDA != 0),
          "public compile-time macro and runtime backend state disagree");

  std::string diagnostic;
  const bool available = meep::gpu::runtime_available(&diagnostic);

  const char *original_backend = std::getenv("MEEP_GPU_BACKEND");
  const bool had_original_backend = original_backend != nullptr;
  const std::string saved_backend = original_backend ? original_backend : "";
  setenv("MEEP_GPU_BACKEND", "invalid-test-value", 1);
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool invalid_environment_threw = false;
    try {
      (void)meep::gpu::detail::cuda_active();
    }
    catch (const std::invalid_argument &) {
      invalid_environment_threw = true;
    }
    require(invalid_environment_threw,
            "invalid lazy MEEP_GPU_BACKEND query must remain failed "
            "transactionally");
  }

  if (!available) {
    setenv("MEEP_GPU_BACKEND", "cuda", 1);
    for (int attempt = 0; attempt < 2; ++attempt) {
      bool unavailable_environment_threw = false;
      try {
        (void)meep::gpu::detail::cuda_active();
      }
      catch (const std::runtime_error &) {
        unavailable_environment_threw = true;
      }
      require(unavailable_environment_threw,
              "failed lazy CUDA query must remain failed transactionally");
    }
  }
  setenv("MEEP_GPU_BACKEND", "cpu", 1);
  require_cuda_active_hot_view(
      false, "successful lazy CPU configuration must publish a false hot view");
  if (had_original_backend)
    setenv("MEEP_GPU_BACKEND", saved_backend.c_str(), 1);
  else
    unsetenv("MEEP_GPU_BACKEND");

#if MEEP_HAVE_CUDA
  require(meep::gpu::compiled_architectures() != "none",
          "CUDA build must expose compiled architectures");
  if (available) {
    const auto devices = meep::gpu::enumerate_devices();
    bool found_compatible = false;
    for (const auto &device : devices)
      found_compatible = found_compatible || device.compatible;
    require(found_compatible, "available CUDA runtime must enumerate a compatible device");
  }
  else {
    require(!diagnostic.empty(), "unavailable CUDA runtime must provide a diagnostic");
  }
#else
  require(!available, "CPU-only build cannot report an available CUDA backend");
  require(!diagnostic.empty(), "CPU-only stub must provide a diagnostic");
  require(meep::gpu::enumerate_devices().empty(), "CPU-only stub must enumerate no devices");
  require(meep::gpu::compiled_architectures() == "none",
          "CPU-only stub must report no compiled architectures");
  bool threw = false;
  try {
    meep::gpu::select_device(0);
  }
  catch (const std::runtime_error &) {
    threw = true;
  }
  require(threw, "CPU-only select_device must fail clearly");
#endif

  meep::gpu::set_backend(meep::gpu::backend_mode::cpu);
  require(meep::gpu::requested_backend() == meep::gpu::backend_mode::cpu,
          "set_backend(cpu) must update the requested backend");
  require(meep::gpu::active_backend() == meep::gpu::backend_mode::cpu,
          "set_backend(cpu) must keep CPU active");
  require_cuda_active_hot_view(
      false, "set_backend(cpu) must publish a false hot view");

#if MEEP_HAVE_CUDA
  const auto require_rejected_cuda_configuration =
      [](const char *name, const char *value, const char *message) {
        const char *original = std::getenv(name);
        const bool had_original = original != nullptr;
        const std::string saved = original ? original : "";
        const std::uint64_t generation_before =
            meep::gpu::detail::backend_generation();
        const std::string diagnostic_before =
            meep::gpu::backend_diagnostic();
        setenv(name, value, 1);
        bool rejected = false;
        try {
          meep::gpu::set_backend(meep::gpu::backend_mode::cuda);
        }
        catch (const std::exception &) {
          rejected = true;
        }
        if (had_original)
          setenv(name, saved.c_str(), 1);
        else
          unsetenv(name);
        require(rejected, message);
        require(
            meep::gpu::requested_backend() ==
                    meep::gpu::backend_mode::cpu &&
                meep::gpu::active_backend() ==
                    meep::gpu::backend_mode::cpu,
            "failed CUDA configuration must preserve the CPU transaction");
        require_cuda_active_hot_view(
            false,
            "failed CUDA configuration must republish the CPU hot view");
        require(meep::gpu::detail::backend_generation() == generation_before &&
                    meep::gpu::backend_diagnostic() == diagnostic_before,
                "failed CUDA configuration must preserve generation and "
                "diagnostic state");
      };
  require_rejected_cuda_configuration(
      "MEEP_GPU_DEVICE", "invalid",
      "CUDA mode must reject a malformed explicit CUDA device");
  require_rejected_cuda_configuration(
      "MEEP_GPU_DEVICE", "2147483647",
      "CUDA mode must reject an unavailable explicit CUDA device");
  require_rejected_cuda_configuration(
      "MEEP_GPU_ALLOW_OVERSUBSCRIBE", "invalid",
      "CUDA mode must reject malformed oversubscription policy");
#endif

  int cpu_pin_owner = 0;
  meep::gpu::detail::pin_backend_for_owner(
      &cpu_pin_owner, meep::gpu::backend_mode::cpu);
  const std::uint64_t cpu_pin_generation =
      meep::gpu::detail::backend_generation();
  const std::string cpu_pin_diagnostic = meep::gpu::backend_diagnostic();
  bool pinned_backend_change_rejected = false;
  try {
    meep::gpu::set_backend(meep::gpu::backend_mode::cuda);
  }
  catch (const std::runtime_error &) {
    pinned_backend_change_rejected = true;
  }
  require(pinned_backend_change_rejected,
          "a CPU-pinned owner must reject a CUDA backend transition");
#if MEEP_HAVE_CUDA
  bool pinned_device_change_rejected = false;
  try {
    meep::gpu::select_device(0);
  }
  catch (const std::runtime_error &) {
    pinned_device_change_rejected = true;
  }
  require(pinned_device_change_rejected,
          "a CPU-pinned owner must reject a CUDA device transition");
#endif
  bool pinned_owner_activation_rejected = false;
  try {
    meep::gpu::detail::activate_backend_for_owner(
        meep::gpu::backend_mode::cuda, "test invalid owner activation");
  }
  catch (const std::logic_error &) {
    pinned_owner_activation_rejected = true;
  }
  require(pinned_owner_activation_rejected,
          "an explicit CPU request must reject owner CUDA activation");
  require_cuda_active_hot_view(
      false, "rejected CPU-pinned transitions must preserve the CPU hot view");
  require(meep::gpu::detail::backend_generation() == cpu_pin_generation &&
              meep::gpu::backend_diagnostic() == cpu_pin_diagnostic,
          "rejected CPU-pinned transitions must preserve generation and "
          "diagnostic state");
  meep::gpu::detail::unpin_backend_for_owner(&cpu_pin_owner);

  meep::gpu::reset_dispatch_statistics();
  const auto empty_statistics = meep::gpu::get_dispatch_statistics();
  require(empty_statistics.cpu_curl_calls == 0 && empty_statistics.cuda_curl_calls == 0,
          "reset dispatch statistics must clear curl counters");
  const auto empty_resident_statistics = meep::gpu::get_resident_statistics();
  require(empty_resident_statistics.host_to_device_bytes_avoided == 0 &&
              empty_resident_statistics.device_to_host_bytes_avoided == 0 &&
              empty_resident_statistics.device_buffer_allocations == 0 &&
              empty_resident_statistics.device_buffer_reuses == 0,
          "reset dispatch statistics must clear resident-cache counters");
  require(meep::gpu::get_live_resident_device_buffers() == 0,
          "backend API test unexpectedly retained a live resident buffer");
  const auto empty_field_update_statistics =
      meep::gpu::get_field_update_statistics();
  require(empty_field_update_statistics.cpu_update_eh_calls == 0 &&
              empty_field_update_statistics.cpu_update_eh_points == 0 &&
              empty_field_update_statistics.cuda_update_eh_calls == 0 &&
              empty_field_update_statistics.cuda_update_eh_points == 0,
          "reset dispatch statistics must clear E/H update counters");
  const auto empty_polarization_statistics =
      meep::gpu::get_polarization_statistics();
  require(empty_polarization_statistics.cpu_update_calls == 0 &&
              empty_polarization_statistics.cpu_update_points == 0 &&
              empty_polarization_statistics.cuda_update_calls == 0 &&
              empty_polarization_statistics.cuda_update_points == 0,
          "reset dispatch statistics must clear polarization counters");
  const auto sparse_generation_before =
      meep::gpu::get_sparse_lorentzian_statistics().reset_generation;
  meep::gpu::reset_sparse_lorentzian_statistics();
  const auto empty_sparse_statistics =
      meep::gpu::get_sparse_lorentzian_statistics();
  require(empty_sparse_statistics.opt_in_calls == 0 &&
              empty_sparse_statistics.plan_build_attempts == 0 &&
              empty_sparse_statistics.device_guard_scan_calls == 0 &&
              empty_sparse_statistics.device_sparse_update_calls == 0 &&
              empty_sparse_statistics.live_plans == 0 &&
              empty_sparse_statistics.live_guards == 0 &&
              empty_sparse_statistics.retained_index_bytes == 0 &&
              empty_sparse_statistics.reset_generation ==
                  sparse_generation_before + 1 &&
              !empty_sparse_statistics.counter_overflow &&
              !empty_sparse_statistics.teardown_incomplete,
          "checked sparse Lorentzian statistics reset must preserve an empty "
          "and monotonic public state");
  const auto empty_source_statistics = meep::gpu::get_source_statistics();
  require(empty_source_statistics.cpu_update_calls == 0 &&
              empty_source_statistics.cpu_update_points == 0 &&
              empty_source_statistics.cuda_update_calls == 0 &&
              empty_source_statistics.cuda_update_points == 0,
          "reset dispatch statistics must clear source counters");
  const auto empty_boundary_statistics =
      meep::gpu::get_boundary_statistics();
  require(empty_boundary_statistics.cpu_update_calls == 0 &&
              empty_boundary_statistics.cpu_update_points == 0 &&
              empty_boundary_statistics.cuda_update_calls == 0 &&
              empty_boundary_statistics.cuda_update_points == 0,
          "reset dispatch statistics must clear boundary counters");
  const auto empty_dft_statistics = meep::gpu::get_dft_statistics();
  require(empty_dft_statistics.cpu_update_calls == 0 &&
              empty_dft_statistics.cpu_update_points == 0 &&
              empty_dft_statistics.cuda_update_calls == 0 &&
              empty_dft_statistics.cuda_update_points == 0,
          "reset dispatch statistics must clear DFT counters");
  const auto empty_runtime_statistics =
      meep::gpu::get_runtime_touch_statistics();
  require(empty_runtime_statistics.availability_probes == 0 &&
              empty_runtime_statistics.device_enumerations == 0 &&
              empty_runtime_statistics.device_selections == 0,
          "reset dispatch statistics must clear CUDA runtime-touch counters");
  const auto empty_mpi_completion_statistics =
      meep::gpu::get_mpi_completion_statistics();
  require(empty_mpi_completion_statistics.waitsome_executions == 0 &&
              empty_mpi_completion_statistics.waitall_executions == 0,
          "reset dispatch statistics must clear MPI completion counters");

  bool automatic_rejected = false;
  try {
    meep::gpu::set_backend(
        static_cast<meep::gpu::backend_mode>(1));
  }
  catch (const std::invalid_argument &) {
    automatic_rejected = true;
  }
  require(automatic_rejected,
          "gpmeep v1.1.0 must reject automatic backend selection");
  require(meep::gpu::requested_backend() == meep::gpu::backend_mode::cpu &&
              meep::gpu::active_backend() == meep::gpu::backend_mode::cpu,
          "rejected automatic selection changed the CPU transaction");
  require_cuda_active_hot_view(
      false, "rejected automatic selection must preserve the CPU hot view");
  require(!meep::gpu::backend_diagnostic().empty(),
          "resolved backend must expose a diagnostic");

  if (available) {
    meep::gpu::set_backend(meep::gpu::backend_mode::cuda);
    require(meep::gpu::active_backend() == meep::gpu::backend_mode::cuda,
            "required CUDA backend must become active");
    require_cuda_active_hot_view(
        true, "set_backend(cuda) must publish a true hot view");

    const int selected_ordinal = meep::gpu::selected_device();
    const std::string selected_identifier =
        meep::gpu::selected_device_identifier();
    const std::uint64_t generation_before_same_device =
        meep::gpu::detail::backend_generation();
    meep::gpu::select_device(selected_ordinal);
    require_cuda_active_hot_view(
        true, "successful CUDA device selection must republish a true hot view");
    require(meep::gpu::selected_device() == selected_ordinal &&
                meep::gpu::selected_device_identifier() == selected_identifier,
            "same-device selection must preserve CUDA identity");
    require(meep::gpu::detail::backend_generation() ==
                generation_before_same_device + 1,
            "successful CUDA device selection must advance generation once");

    const std::uint64_t generation_before_invalid_device =
        meep::gpu::detail::backend_generation();
    const std::string diagnostic_before_invalid_device =
        meep::gpu::backend_diagnostic();
    bool invalid_device_rejected = false;
    try {
      meep::gpu::select_device(std::numeric_limits<int>::max());
    }
    catch (const std::exception &) {
      invalid_device_rejected = true;
    }
    require(invalid_device_rejected,
            "invalid CUDA device selection must fail transactionally");
    require_cuda_active_hot_view(
        true, "failed CUDA device selection must republish a true hot view");
    require(meep::gpu::selected_device() == selected_ordinal &&
                meep::gpu::selected_device_identifier() == selected_identifier,
            "failed CUDA device selection must preserve CUDA identity");
    require(meep::gpu::detail::backend_generation() ==
                    generation_before_invalid_device &&
                meep::gpu::backend_diagnostic() ==
                    diagnostic_before_invalid_device,
            "failed CUDA device selection must preserve generation and "
            "diagnostic state");

    const char *original_device = std::getenv("MEEP_GPU_DEVICE");
    const bool had_original_device = original_device != nullptr;
    const std::string saved_device = original_device ? original_device : "";
    std::string equivalent_device = "0" + std::to_string(selected_ordinal);
    if (had_original_device && equivalent_device == saved_device)
      equivalent_device.insert(equivalent_device.begin(), '0');
    setenv("MEEP_GPU_DEVICE", equivalent_device.c_str(), 1);
    auto refresh_status =
        meep::gpu::detail::probe_backend_for_distributed_step();
    require(refresh_status.configuration_ok && refresh_status.cuda_active &&
                refresh_status.cuda_required,
            "equivalent CUDA environment refresh must succeed");
    require_cuda_active_hot_view(
        true, "successful CUDA environment refresh must publish a true hot view");

    const std::uint64_t generation_before_invalid_refresh =
        meep::gpu::detail::backend_generation();
    const std::string diagnostic_before_invalid_refresh =
        meep::gpu::backend_diagnostic();
    setenv("MEEP_GPU_DEVICE", "invalid", 1);
    refresh_status = meep::gpu::detail::probe_backend_for_distributed_step();
    require(!refresh_status.configuration_ok,
            "malformed CUDA environment refresh must fail");
    require_cuda_active_hot_view(
        true, "failed CUDA environment refresh must republish the old hot view");
    require(meep::gpu::selected_device() == selected_ordinal &&
                meep::gpu::selected_device_identifier() == selected_identifier,
            "failed CUDA environment refresh must preserve CUDA identity");
    require(meep::gpu::detail::backend_generation() ==
                    generation_before_invalid_refresh &&
                meep::gpu::backend_diagnostic() ==
                    diagnostic_before_invalid_refresh,
            "failed CUDA environment refresh must preserve generation and "
            "diagnostic state");

    setenv("MEEP_GPU_DEVICE", equivalent_device.c_str(), 1);
    refresh_status = meep::gpu::detail::probe_backend_for_distributed_step();
    require(refresh_status.configuration_ok && refresh_status.cuda_active,
            "restoring the valid CUDA environment must recover the probe");
    if (had_original_device)
      setenv("MEEP_GPU_DEVICE", saved_device.c_str(), 1);
    else
      unsetenv("MEEP_GPU_DEVICE");
    refresh_status = meep::gpu::detail::probe_backend_for_distributed_step();
    require(refresh_status.configuration_ok && refresh_status.cuda_active,
            "restoring the caller CUDA environment must refresh successfully");
    require_cuda_active_hot_view(
        true, "restored CUDA environment must retain a true hot view");

    int cuda_pin_owner = 0;
    meep::gpu::detail::pin_backend_for_owner(
        &cuda_pin_owner, meep::gpu::backend_mode::cuda);
    const std::uint64_t cuda_pin_generation =
        meep::gpu::detail::backend_generation();
    const std::string cuda_pin_diagnostic = meep::gpu::backend_diagnostic();
    bool pinned_cpu_change_rejected = false;
    try {
      meep::gpu::set_backend(meep::gpu::backend_mode::cpu);
    }
    catch (const std::runtime_error &) {
      pinned_cpu_change_rejected = true;
    }
    require(pinned_cpu_change_rejected,
            "a CUDA-pinned owner must reject a CPU backend transition");
    bool pinned_invalid_device_rejected = false;
    try {
      meep::gpu::select_device(std::numeric_limits<int>::max());
    }
    catch (const std::runtime_error &) {
      pinned_invalid_device_rejected = true;
    }
    require(pinned_invalid_device_rejected,
            "a CUDA-pinned owner must reject a device transition");
    bool pinned_cpu_activation_rejected = false;
    try {
      meep::gpu::detail::activate_backend_for_owner(
          meep::gpu::backend_mode::cpu, "test invalid owner activation");
    }
    catch (const std::logic_error &) {
      pinned_cpu_activation_rejected = true;
    }
    require(pinned_cpu_activation_rejected,
            "an explicit CUDA request must reject owner CPU activation");
    require_cuda_active_hot_view(
        true, "rejected CUDA-pinned transitions must preserve the CUDA hot view");
    require(meep::gpu::detail::backend_generation() == cuda_pin_generation &&
                meep::gpu::backend_diagnostic() == cuda_pin_diagnostic,
            "rejected CUDA-pinned transitions must preserve generation and "
            "diagnostic state");
    meep::gpu::detail::unpin_backend_for_owner(&cuda_pin_owner);

    const std::uint64_t generation_before_cuda_activation =
        meep::gpu::detail::backend_generation();
    meep::gpu::detail::activate_backend_for_owner(
        meep::gpu::backend_mode::cuda, "test valid CUDA owner activation");
    require_cuda_active_hot_view(
        true, "valid CUDA owner activation must retain the CUDA hot view");
    require(meep::gpu::detail::backend_generation() ==
                generation_before_cuda_activation,
            "diagnostic-only CUDA owner activation must not advance generation");
  }
  else {
    bool threw = false;
    try {
      meep::gpu::set_backend(meep::gpu::backend_mode::cuda);
    }
    catch (const std::runtime_error &) {
      threw = true;
    }
    require(threw, "required CUDA backend must reject an unavailable runtime");
    require(meep::gpu::requested_backend() == meep::gpu::backend_mode::cpu,
            "failed backend selection must preserve the previous configuration");
    require_cuda_active_hot_view(
        false, "failed unavailable CUDA selection must preserve the CPU hot view");
  }
  meep::gpu::set_backend(meep::gpu::backend_mode::cpu);
  const std::uint64_t generation_before_cpu_activation =
      meep::gpu::detail::backend_generation();
  meep::gpu::detail::activate_backend_for_owner(
      meep::gpu::backend_mode::cpu, "test valid CPU owner activation");
  require_cuda_active_hot_view(
      false, "final CPU activation must publish a false hot view");
  require(meep::gpu::detail::backend_generation() ==
              generation_before_cpu_activation,
          "diagnostic-only CPU owner activation must not advance generation");

  std::cout << "PASS: Meep GPU backend API compiled=" << meep::gpu::backend_compiled()
            << " runtime_available=" << available
            << " architectures=" << meep::gpu::compiled_architectures() << '\n';
  return 0;
}
