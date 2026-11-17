#!/bin/bash -p
set -euo pipefail

if [[ "$-" != *p* ]]; then
  echo "error: upstream baseline builder requires protected Bash mode" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREFIX="${REPO_ROOT}/.envs/meep-gpu-cuda-mpi"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_SHA256="9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_CACHE="${MAMBA_ROOT}/cache"
CONTRACT="${REPO_ROOT}/packaging/v110-upstream-cpu-baseline.json"
DEPENDENCY_LOCK="${REPO_ROOT}/environment/locks/cuda-mpi-linux-64.lock"
BUILD_DIR="${REPO_ROOT}/build/upstream-meep-cpu-mpi-python-fp32"
INSTALL_DIR="${REPO_ROOT}/install/upstream-meep-cpu-mpi-python-fp32"
RECEIPT_TOOL="${SCRIPT_DIR}/upstream-cpu-baseline-receipt.py"
ARCHIVE_SHA256="a9e92dbdf820820b341965c368a5e8c34241e79a142318421da1ed0751527202"

verify_micromamba_binary() {
  if [[ ! -f "${MAMBA}" || -L "${MAMBA}" || ! -x "${MAMBA}" ]] || \
     [[ "$(/usr/bin/realpath -e -- "${MAMBA}")" != "${MAMBA}" ]] || \
     [[ "$(/usr/bin/sha256sum "${MAMBA}" | /usr/bin/awk '{print $1}')" != \
        "${MAMBA_SHA256}" ]]; then
    echo "error: Micromamba does not match the pinned protected binary" >&2
    exit 1
  fi
}

STAGE=bootstrap
if [[ "${1:-}" == worker ]]; then
  STAGE=worker
  shift
fi

if [[ "${STAGE}" == bootstrap ]]; then
  if [[ "$#" -ne 2 || "$1" != --source-archive ]]; then
    echo "usage: $0 --source-archive PATH" >&2
    exit 2
  fi
  SOURCE_ARCHIVE="$(/usr/bin/realpath -e -- "$2")"
  if [[ "$(/usr/bin/sha256sum "${SOURCE_ARCHIVE}" | /usr/bin/awk '{print $1}')" != "${ARCHIVE_SHA256}" ]]; then
    echo "error: upstream source archive SHA-256 differs" >&2
    exit 1
  fi
  verify_micromamba_binary
  if [[ ! -x "${PREFIX}/bin/python" ]]; then
    echo "error: exact-lock gpmeep build environment is unavailable" >&2
    exit 1
  fi
  /usr/bin/mkdir -p "${MAMBA_CACHE}"
  if [[ -L "${MAMBA_CACHE}" || ! -d "${MAMBA_CACHE}" ]]; then
    echo "error: repository-local micromamba cache is unavailable" >&2
    exit 1
  fi
  USER_HOME="$(/usr/bin/getent passwd "$(/usr/bin/id -u)" | /usr/bin/cut -d: -f6)"
  exec /usr/bin/env -i HOME="${USER_HOME}" \
    XDG_CACHE_HOME="${MAMBA_CACHE}" PATH=/usr/bin:/bin \
    "${MAMBA}" --no-rc run --clean-env \
    --env "HOME=${USER_HOME}" \
    --env "XDG_CACHE_HOME=${MAMBA_CACHE}" \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONPYCACHEPREFIX=/dev/null \
    --env MPLBACKEND=Agg \
    --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" \
    /bin/bash -p "${BASH_SOURCE[0]}" worker --source-archive "${SOURCE_ARCHIVE}"
fi

if [[ "${STAGE}" != worker || "$#" -ne 2 || "$1" != --source-archive ]]; then
  echo "usage: $0 --source-archive PATH" >&2
  exit 2
fi
SOURCE_ARCHIVE="$(/usr/bin/realpath -e -- "$2")"
verify_micromamba_binary
if [[ "${CONDA_PREFIX:-}" != "${PREFIX}" ]]; then
  echo "error: upstream worker escaped the exact-lock environment" >&2
  exit 1
fi
if [[ "$(/usr/bin/sha256sum "${SOURCE_ARCHIVE}" | /usr/bin/awk '{print $1}')" != "${ARCHIVE_SHA256}" ]]; then
  echo "error: upstream source archive SHA-256 changed" >&2
  exit 1
fi
for tool in autoreconf make mpicxx mpiexec h5pcc python swig unzip; do
  TOOL_PATH="$(command -v "${tool}")"
  if [[ "${tool}" != unzip && "${TOOL_PATH}" != "${PREFIX}/bin/"* ]]; then
    echo "error: ${tool} resolved outside the exact-lock environment: ${TOOL_PATH}" >&2
    exit 1
  fi
done

PREFIX_FONTCONFIG_CACHE="${PREFIX}/var/cache/fontconfig"
assert_prefix_fontconfig_cache_sealed() {
  if [[ ! -d "${PREFIX_FONTCONFIG_CACHE}" || \
        -L "${PREFIX_FONTCONFIG_CACHE}" || \
        "$(/usr/bin/stat -c '%a' "${PREFIX_FONTCONFIG_CACHE}")" != 555 || \
        ! -f "${PREFIX_FONTCONFIG_CACHE}/.leave" || \
        -L "${PREFIX_FONTCONFIG_CACHE}/.leave" || \
        "$(/usr/bin/stat -c '%a' "${PREFIX_FONTCONFIG_CACHE}/.leave")" != 444 ]]; then
    echo "error: exact-lock Fontconfig cache is not sealed" >&2
    exit 1
  fi
  PREFIX_FONTCONFIG_UNEXPECTED="$(
    /usr/bin/find "${PREFIX_FONTCONFIG_CACHE}" -mindepth 1 -maxdepth 1 \
      ! -name .leave -print -quit
  )"
  if [[ -n "${PREFIX_FONTCONFIG_UNEXPECTED}" ]]; then
    echo "error: exact-lock Fontconfig cache contains runtime state" >&2
    exit 1
  fi
}
assert_prefix_fontconfig_cache_sealed

PACKAGE_SET_WORK="$(/usr/bin/mktemp -d "${MAMBA_CACHE}/upstream-package-set.XXXXXXXX")"
cleanup_package_set_work() {
  /usr/bin/rm -f -- \
    "${PREFIX_PACKAGE_SET:-}" "${LOCK_PACKAGE_SET:-}" \
    "${POST_BUILD_PACKAGE_SET:-}" "${ENVIRONMENT_PREFLIGHT:-}" \
    "${ENVIRONMENT_CAPTURE:-}"
  /usr/bin/rmdir -- "${PACKAGE_SET_WORK}" 2>/dev/null || true
}
trap cleanup_package_set_work EXIT
PREFIX_PACKAGE_SET="${PACKAGE_SET_WORK}/prefix.txt"
LOCK_PACKAGE_SET="${PACKAGE_SET_WORK}/lock.txt"
POST_BUILD_PACKAGE_SET="${PACKAGE_SET_WORK}/post-build-prefix.txt"
ENVIRONMENT_PREFLIGHT="${PACKAGE_SET_WORK}/environment-preflight.json"
ENVIRONMENT_CAPTURE="${PACKAGE_SET_WORK}/environment-capture.json"
capture_prefix_package_set() {
  local output="$1"
  "${MAMBA}" --no-rc list --root-prefix "${MAMBA_ROOT}" \
    --prefix "${PREFIX}" --explicit --sha256 \
    | /usr/bin/sed -n '/:\/\//p' | /usr/bin/sort >"${output}"
  if [[ ! -s "${output}" ]]; then
    echo "error: exact-lock environment package set is empty" >&2
    exit 1
  fi
}
capture_prefix_package_set "${PREFIX_PACKAGE_SET}"
/usr/bin/python3 -I -S "${RECEIPT_TOOL}" verify-environment \
  --repo "${REPO_ROOT}" >"${ENVIRONMENT_PREFLIGHT}"
/usr/bin/sed -n '/:\/\//p' "${DEPENDENCY_LOCK}" \
  | /usr/bin/sort >"${LOCK_PACKAGE_SET}"
CONTRACT_LOCK_SHA256="$(
  "${PREFIX}/bin/python" -I -c \
    'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["dependency_environment"]["lock_sha256"])' \
    "${CONTRACT}"
)"
LOCK_SHA256="$(/usr/bin/sha256sum "${DEPENDENCY_LOCK}" | /usr/bin/awk '{print $1}')"
if [[ ! "${CONTRACT_LOCK_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
   [[ "${LOCK_SHA256}" != "${CONTRACT_LOCK_SHA256}" ]]; then
  echo "error: dependency lock digest differs from upstream baseline contract" >&2
  exit 1
fi
if [[ ! -s "${LOCK_PACKAGE_SET}" ]] || \
   ! /usr/bin/cmp -s "${PREFIX_PACKAGE_SET}" "${LOCK_PACKAGE_SET}"; then
  echo "error: exact-lock environment package set differs from dependency lock" >&2
  /usr/bin/diff -u "${LOCK_PACKAGE_SET}" "${PREFIX_PACKAGE_SET}" >&2 || true
  exit 1
fi

TIMESTAMP="$(/usr/bin/date -u +%Y%m%dT%H%M%SZ).$$"
for directory in "${BUILD_DIR}" "${INSTALL_DIR}"; do
  if [[ -e "${directory}" || -L "${directory}" ]]; then
    /usr/bin/mv -- "${directory}" "${directory}.previous.${TIMESTAMP}"
  fi
done
/usr/bin/mkdir -p \
  "${BUILD_DIR}/source" "${BUILD_DIR}/qualification-logs" "${INSTALL_DIR}"
QUALIFICATION_LOGS="${BUILD_DIR}/qualification-logs"
/usr/bin/unzip -q "${SOURCE_ARCHIVE}" -d "${BUILD_DIR}/source"
SOURCE_ROOT="${BUILD_DIR}/source/meep-master"
if [[ ! -d "${SOURCE_ROOT}" || -L "${SOURCE_ROOT}" ]]; then
  echo "error: contracted upstream source root is absent" >&2
  exit 1
fi
UNSAFE_ENTRY="$(/usr/bin/find "${BUILD_DIR}/source" \( -type l -o \( ! -type d ! -type f \) \) -print -quit)"
if [[ -n "${UNSAFE_ENTRY}" ]]; then
  echo "error: upstream archive extracted an unsafe entry: ${UNSAFE_ENTRY}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/dev/null
export OMP_NUM_THREADS=1
QUALIFICATION_CACHE="${BUILD_DIR}/qualification-cache"
QUALIFICATION_FONTCONFIG="${BUILD_DIR}/qualification-fontconfig.conf"
/usr/bin/mkdir -p \
  "${QUALIFICATION_CACHE}/fontconfig" "${QUALIFICATION_CACHE}/matplotlib"
# Do not include PREFIX/etc/fonts/fonts.conf: it contains an absolute cachedir
# under the exact-lock prefix and lets a read-only qualification mutate it.
{
  printf '%s\n' '<?xml version="1.0"?>'
  printf '%s\n' '<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">'
  printf '%s\n' '<fontconfig>'
  printf '%s\n' '  <dir>/usr/share/fonts</dir>'
  printf '  <dir>%s/share/fonts</dir>\n' "${PREFIX}"
  printf '  <dir>%s/fonts</dir>\n' "${PREFIX}"
  printf '  <include ignore_missing="yes">%s/etc/fonts/conf.d</include>\n' "${PREFIX}"
  printf '  <cachedir>%s/fontconfig</cachedir>\n' "${QUALIFICATION_CACHE}"
  printf '%s\n' '</fontconfig>'
} >"${QUALIFICATION_FONTCONFIG}"
export XDG_CACHE_HOME="${QUALIFICATION_CACHE}"
export MPLCONFIGDIR="${QUALIFICATION_CACHE}/matplotlib"
export FONTCONFIG_FILE="${QUALIFICATION_FONTCONFIG}"
MAKE_JOBS="${MEEP_GPU_MAKE_JOBS:-8}"

cd "${SOURCE_ROOT}"
/usr/bin/timeout --kill-after=30s 600s autoreconf --verbose --install --symlink --force
CONFIGURE_ARGUMENTS=(
  --enable-maintainer-mode
  --enable-shared
  --enable-single
  --with-openmp
  --with-mpi
  --with-python
  --without-scheme
  "--prefix=${INSTALL_DIR}"
)
MPICXX="${PREFIX}/bin/mpicxx" \
  /usr/bin/timeout --kill-after=30s 600s \
  "${SOURCE_ROOT}/configure" "${CONFIGURE_ARGUMENTS[@]}"
CXXFLAGS_LINE="$(/usr/bin/grep -E '^CXXFLAGS = ' Makefile)"
if [[ ! "${CXXFLAGS_LINE}" =~ (^|[[:space:]])-O2([[:space:]]|$) ]]; then
  echo "error: upstream CPU baseline was not configured with contracted -O2" >&2
  exit 1
fi
/usr/bin/timeout --kill-after=60s 3600s make -j"${MAKE_JOBS}"
QUALIFICATION_HOME="${BUILD_DIR}/qualification-home"
/usr/bin/install -d -m 700 "${QUALIFICATION_HOME}"
/usr/bin/timeout --kill-after=30s 600s \
  make -C python -j1 check TESTS=tests/test_binary_partition_utils.py \
  >"${QUALIFICATION_LOGS}/upstream-binary-partition-utils.log" 2>&1
INSTALL_PYCACHE="${BUILD_DIR}/install-pycache"
/usr/bin/install -d -m 700 "${INSTALL_PYCACHE}"
/usr/bin/env PYTHONPYCACHEPREFIX="${INSTALL_PYCACHE}" \
  /usr/bin/timeout --kill-after=60s 1800s make install

PYTHON_ABI="$("${PREFIX}/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
INSTALLED_PYTHON="${INSTALL_DIR}/lib/${PYTHON_ABI}/site-packages"
LIBMEEP="$(/usr/bin/realpath -e "${INSTALL_DIR}/lib/libmeep.so")"
PYTHON_EXTENSION="$(/usr/bin/realpath -e "${INSTALLED_PYTHON}/meep/_meep.so")"
MPB_EXTENSION="$(/usr/bin/realpath -e "${INSTALLED_PYTHON}/meep/mpb/_mpb.so")"
LIBPYMPB="$(/usr/bin/realpath -e "${INSTALL_DIR}/lib/libpympb.so")"

/usr/bin/env -i HOME="${QUALIFICATION_HOME}" \
  PATH="${PREFIX}/bin:/usr/bin:/bin" \
  PYTHONPATH="${INSTALLED_PYTHON}" \
  LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${PREFIX}/lib" \
  XDG_CACHE_HOME="${QUALIFICATION_CACHE}" \
  MPLCONFIGDIR="${QUALIFICATION_CACHE}/matplotlib" \
  FONTCONFIG_FILE="${QUALIFICATION_FONTCONFIG}" \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "${PREFIX}/bin/python" -c \
  'import meep as mp; assert mp.is_single_precision(); assert mp.with_mpi(); assert not hasattr(mp, "gpu"); print(f"upstream-single-import:PASS version={mp.__version__}")' \
  >"${QUALIFICATION_LOGS}/python-single-process-import.log" 2>&1

/usr/bin/env -i HOME="${QUALIFICATION_HOME}" \
  PATH="${PREFIX}/bin:/usr/bin:/bin" \
  PYTHONPATH="${INSTALLED_PYTHON}" \
  LD_LIBRARY_PATH="${INSTALL_DIR}/lib:${PREFIX}/lib" \
  XDG_CACHE_HOME="${QUALIFICATION_CACHE}" \
  MPLCONFIGDIR="${QUALIFICATION_CACHE}/matplotlib" \
  FONTCONFIG_FILE="${QUALIFICATION_FONTCONFIG}" \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
  "${PREFIX}/bin/mpiexec" --bind-to core --map-by core:PE=1 -n 2 \
  "${PREFIX}/bin/python" -c \
  'import meep as mp; assert mp.is_single_precision(); assert mp.with_mpi(); assert mp.count_processors() == 2; sim = mp.Simulation(cell_size=mp.Vector3(2, 2), resolution=5); sim.run(until=2); assert sim.timestep() > 0; print("upstream-mpi-smoke:PASS")' \
  >"${QUALIFICATION_LOGS}/python-mpi-two-rank-smoke.log" 2>&1

capture_prefix_package_set "${POST_BUILD_PACKAGE_SET}"
verify_micromamba_binary
if ! /usr/bin/cmp -s "${PREFIX_PACKAGE_SET}" "${POST_BUILD_PACKAGE_SET}" || \
   ! /usr/bin/cmp -s "${LOCK_PACKAGE_SET}" "${POST_BUILD_PACKAGE_SET}"; then
  echo "error: exact-lock environment package set changed during upstream build" >&2
  /usr/bin/diff -u "${PREFIX_PACKAGE_SET}" "${POST_BUILD_PACKAGE_SET}" >&2 || true
  exit 1
fi
/usr/bin/python3 -I -S "${RECEIPT_TOOL}" capture-environment \
  --repo "${REPO_ROOT}" \
  --output "${QUALIFICATION_LOGS}/upstream-dependency-package-set.log" \
  >"${ENVIRONMENT_CAPTURE}"

RECEIPT="${BUILD_DIR}/upstream-cpu-baseline-receipt.json"
"${PREFIX}/bin/python" "${RECEIPT_TOOL}" create \
  --repo "${REPO_ROOT}" --archive "${SOURCE_ARCHIVE}" \
  --python "${PREFIX}/bin/python" --mpiexec "${PREFIX}/bin/mpiexec" \
  --artifact "libmeep=${LIBMEEP}" \
  --artifact "python_extension=${PYTHON_EXTENSION}" \
  --artifact "mpb_extension=${MPB_EXTENSION}" \
  --artifact "libpympb=${LIBPYMPB}" \
  --qualification "python_mpi_two_rank_smoke=${QUALIFICATION_LOGS}/python-mpi-two-rank-smoke.log" \
  --qualification "python_single_process_import=${QUALIFICATION_LOGS}/python-single-process-import.log" \
  --qualification "upstream_binary_partition_utils=${QUALIFICATION_LOGS}/upstream-binary-partition-utils.log" \
  --qualification "upstream_build_configuration=${SOURCE_ROOT}/Makefile" \
  --qualification "upstream_dependency_package_set=${QUALIFICATION_LOGS}/upstream-dependency-package-set.log" \
  --qualification "upstream_fontconfig_configuration=${QUALIFICATION_FONTCONFIG}" \
  --output "${RECEIPT}"
"${PREFIX}/bin/python" "${RECEIPT_TOOL}" verify \
  --repo "${REPO_ROOT}" --receipt "${RECEIPT}"
assert_prefix_fontconfig_cache_sealed
echo "upstream Meep CPU/MPI/Python FP32 baseline installed in ${INSTALL_DIR}"
