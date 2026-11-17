#!/bin/bash -p
set -euo pipefail

if [[ "$-" != *p* ]]; then
  echo "error: package builder requires protected Bash mode" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$('/usr/bin/dirname' "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAMBA="${REPO_ROOT}/.tools/micromamba"
MAMBA_ROOT="${REPO_ROOT}/.micromamba"
MAMBA_PROCESS_CACHE="${MAMBA_ROOT}/package-builder-xdg-cache"
PREFIX="${REPO_ROOT}/.envs/gpmeep-package-builder"
ENVIRONMENT_PROFILE="package-builder"
if [[ -f "${REPO_ROOT}/packaging/v110-development-contract.json" ]]; then
  PREFIX="${REPO_ROOT}/.envs/gpmeep-v1.1-builder"
  ENVIRONMENT_PROFILE="v110-builder"
fi
LOCK="${REPO_ROOT}/environment/locks/package-builder-linux-64.lock"
OUTPUT_DIRECTORY="${REPO_ROOT}/dist"
CUDA_ARCHITECTURES="AUTO"
BUILD_JOBS=4
DISTRIBUTION_VERSION="1.1.0"
BUILD_MODE=""
CANDIDATE_EVIDENCE_ROOT=""
CANDIDATE_ATTESTATION=""
CANDIDATE_ATTESTATION_SHA256=""
M4_CANDIDATE_MANIFEST=""
SOURCE_TRANSITION_POLICY=""
SOURCE_TRANSITION_POLICY_SHA256=""
SOURCE_TRANSITION_VERIFIER=""
SOURCE_TRANSITION_VERIFIER_SHA256=""
CANDIDATE_INVERSION_TERMINAL=""
CANDIDATE_INVERSION_TERMINAL_SHA256=""
CANDIDATE_INVERSION_TERMINAL_ID=""
CANDIDATE_INVERSION_RECORDS_DIRECTORY=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output-dir)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --output-dir requires a nonempty path" >&2
        exit 2
      fi
      OUTPUT_DIRECTORY="$2"
      shift 2
      ;;
    --cuda-architectures)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --cuda-architectures requires AUTO or a nonempty LIST" >&2
        exit 2
      fi
      CUDA_ARCHITECTURES="$2"
      shift 2
      ;;
    --jobs)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --jobs requires a positive integer" >&2
        exit 2
      fi
      BUILD_JOBS="$2"
      shift 2
      ;;
    --candidate-evidence-root)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --candidate-evidence-root requires a nonempty path" >&2
        exit 2
      fi
      CANDIDATE_EVIDENCE_ROOT="$2"
      shift 2
      ;;
    --candidate-attestation)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --candidate-attestation requires a nonempty path" >&2
        exit 2
      fi
      CANDIDATE_ATTESTATION="$2"
      shift 2
      ;;
    --candidate-attestation-sha256)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --candidate-attestation-sha256 requires 64 lowercase hex" >&2
        exit 2
      fi
      CANDIDATE_ATTESTATION_SHA256="$2"
      shift 2
      ;;
    --m4-candidate-manifest)
      if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "error: --m4-candidate-manifest requires a nonempty path" >&2
        exit 2
      fi
      M4_CANDIDATE_MANIFEST="$2"
      shift 2
      ;;
    --source-transition-policy)
      [[ "$#" -ge 2 && -n "$2" ]] || { echo "error: --source-transition-policy requires a nonempty path" >&2; exit 2; }
      SOURCE_TRANSITION_POLICY="$2"
      shift 2
      ;;
    --source-transition-policy-sha256)
      [[ "$#" -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || { echo "error: --source-transition-policy-sha256 requires 64 lowercase hex" >&2; exit 2; }
      SOURCE_TRANSITION_POLICY_SHA256="$2"
      shift 2
      ;;
    --source-transition-verifier)
      [[ "$#" -ge 2 && -n "$2" ]] || { echo "error: --source-transition-verifier requires a nonempty path" >&2; exit 2; }
      SOURCE_TRANSITION_VERIFIER="$2"
      shift 2
      ;;
    --source-transition-verifier-sha256)
      [[ "$#" -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || { echo "error: --source-transition-verifier-sha256 requires 64 lowercase hex" >&2; exit 2; }
      SOURCE_TRANSITION_VERIFIER_SHA256="$2"
      shift 2
      ;;
    --candidate-inversion-terminal)
      [[ "$#" -ge 2 && -n "$2" ]] || { echo "error: --candidate-inversion-terminal requires a nonempty absolute path" >&2; exit 2; }
      CANDIDATE_INVERSION_TERMINAL="$2"
      shift 2
      ;;
    --candidate-inversion-terminal-sha256)
      [[ "$#" -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || { echo "error: --candidate-inversion-terminal-sha256 requires 64 lowercase hex" >&2; exit 2; }
      CANDIDATE_INVERSION_TERMINAL_SHA256="$2"
      shift 2
      ;;
    --candidate-inversion-terminal-id)
      [[ "$#" -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || { echo "error: --candidate-inversion-terminal-id requires 64 lowercase hex" >&2; exit 2; }
      CANDIDATE_INVERSION_TERMINAL_ID="$2"
      shift 2
      ;;
    --candidate-inversion-records-directory)
      [[ "$#" -ge 2 && -n "$2" ]] || { echo "error: --candidate-inversion-records-directory requires a nonempty absolute path" >&2; exit 2; }
      CANDIDATE_INVERSION_RECORDS_DIRECTORY="$2"
      shift 2
      ;;
    --development-build)
      if [[ -n "${BUILD_MODE}" ]]; then
        echo "error: build modes are mutually exclusive" >&2
        exit 2
      fi
      BUILD_MODE="development"
      shift
      ;;
    --candidate-build)
      if [[ -n "${BUILD_MODE}" ]]; then
        echo "error: build modes are mutually exclusive" >&2
        exit 2
      fi
      BUILD_MODE="candidate"
      shift
      ;;
    --release-build|--m3-post-audit-release|--release-evidence-root|--release-attestation|--release-attestation-sha256)
      echo "error: M3/release pre-build authority is retired; use --candidate-build with M4 candidate authority" >&2
      exit 2
      ;;
    --help|-h)
      echo "usage: $0 (--development-build | --candidate-build) [--output-dir PATH] [--cuda-architectures AUTO|LIST] [--jobs N] [--candidate-evidence-root PATH --candidate-attestation PATH --candidate-attestation-sha256 HEX --m4-candidate-manifest PATH --source-transition-policy PATH --source-transition-policy-sha256 HEX --source-transition-verifier PATH --source-transition-verifier-sha256 HEX --candidate-inversion-terminal ABSOLUTE_FILE --candidate-inversion-terminal-sha256 HEX --candidate-inversion-terminal-id HEX --candidate-inversion-records-directory ABSOLUTE_DIRECTORY]" >&2
      exit 0
      ;;
    *)
      echo "usage: $0 (--development-build | --candidate-build) [--output-dir PATH] [--cuda-architectures AUTO|LIST] [--jobs N] [--candidate-evidence-root PATH --candidate-attestation PATH --candidate-attestation-sha256 HEX --m4-candidate-manifest PATH --source-transition-policy PATH --source-transition-policy-sha256 HEX --source-transition-verifier PATH --source-transition-verifier-sha256 HEX --candidate-inversion-terminal ABSOLUTE_FILE --candidate-inversion-terminal-sha256 HEX --candidate-inversion-terminal-id HEX --candidate-inversion-records-directory ABSOLUTE_DIRECTORY]" >&2
      exit 2
      ;;
  esac
done
OUTPUT_DIRECTORY="$(/usr/bin/realpath --canonicalize-missing -- "${OUTPUT_DIRECTORY}")"
if [[ "${OUTPUT_DIRECTORY}" != /* ]]; then
  echo "error: canonical output directory is not absolute" >&2
  exit 2
fi
if [[ "$('/usr/bin/uname' -s)" != Linux || "$('/usr/bin/uname' -m)" != x86_64 ]]; then
  echo "error: gpmeep v1 package builds support Linux x86_64" >&2
  exit 2
fi
if [[ ! -f "${LOCK}" ]]; then
  echo "error: exact package-builder lock is absent: ${LOCK}" >&2
  exit 1
fi
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --jobs must be a positive integer" >&2
  exit 2
fi
V110_CONTRACT_VALIDATOR="${SCRIPT_DIR}/validate-v110-development-contract.py"
V110_REGISTRY_VALIDATOR="${SCRIPT_DIR}/validate-v110-inversion-registry.py"
V110_REGISTRY_CLOSER="${SCRIPT_DIR}/close-v110-inversion-registry.py"
V110_ATTESTER="${SCRIPT_DIR}/attest-v110-m0.py"
V110_RECEIPT_TOOL="${SCRIPT_DIR}/v110-release-gate-receipt.py"
V110_RUNTIME_VERIFIER="${SCRIPT_DIR}/verify-v110-final-runtime.py"

canonical_direct_file() {
  local label="$1" input="$2" lexical resolved
  [[ "${input}" == /* ]] || { echo "error: ${label} must be an absolute path" >&2; return 2; }
  lexical="$(/usr/bin/realpath --canonicalize-missing --no-symlinks -- "${input}")"
  [[ -f "${lexical}" && ! -L "${lexical}" ]] || { echo "error: ${label} is not a direct regular file" >&2; return 2; }
  resolved="$(/usr/bin/realpath -- "${lexical}")"
  [[ "${resolved}" == "${lexical}" ]] || { echo "error: ${label} path contains a symlink" >&2; return 2; }
  /usr/bin/printf '%s' "${resolved}"
}

canonical_direct_directory() {
  local label="$1" input="$2" lexical resolved
  [[ "${input}" == /* ]] || { echo "error: ${label} must be an absolute path" >&2; return 2; }
  lexical="$(/usr/bin/realpath --canonicalize-missing --no-symlinks -- "${input}")"
  [[ -d "${lexical}" && ! -L "${lexical}" ]] || { echo "error: ${label} is not a direct directory" >&2; return 2; }
  resolved="$(/usr/bin/realpath -- "${lexical}")"
  [[ "${resolved}" == "${lexical}" ]] || { echo "error: ${label} path contains a symlink" >&2; return 2; }
  /usr/bin/printf '%s' "${resolved}"
}
if [[ -f "${REPO_ROOT}/packaging/v110-development-contract.json" ]]; then
  if [[ -z "${BUILD_MODE}" ]]; then
    echo "error: v1.1.0 requires explicit --development-build or --candidate-build" >&2
    exit 2
  fi
  if [[ "${BUILD_MODE}" == "candidate" ]]; then
    if [[ -z "${CANDIDATE_EVIDENCE_ROOT}" || -z "${CANDIDATE_ATTESTATION}" || -z "${CANDIDATE_ATTESTATION_SHA256}" || -z "${M4_CANDIDATE_MANIFEST}" || -z "${SOURCE_TRANSITION_POLICY}" || -z "${SOURCE_TRANSITION_POLICY_SHA256}" || -z "${SOURCE_TRANSITION_VERIFIER}" || -z "${SOURCE_TRANSITION_VERIFIER_SHA256}" || -z "${CANDIDATE_INVERSION_TERMINAL}" || -z "${CANDIDATE_INVERSION_TERMINAL_SHA256}" || -z "${CANDIDATE_INVERSION_TERMINAL_ID}" || -z "${CANDIDATE_INVERSION_RECORDS_DIRECTORY}" ]]; then
      echo "error: candidate build requires evidence root, attestation, exact attestation SHA-256, M4 candidate manifest, independently pinned source-transition policy, and exact closed inversion terminal/records authority" >&2
      exit 2
    fi
    if [[ ! "${CANDIDATE_ATTESTATION_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "error: --candidate-attestation-sha256 must be 64 lowercase hex" >&2
      exit 2
    fi
    CANDIDATE_EVIDENCE_ROOT="$(canonical_direct_directory "candidate evidence root" "${CANDIDATE_EVIDENCE_ROOT}")"
    CANDIDATE_INVERSION_TERMINAL="$(canonical_direct_file "candidate inversion terminal" "${CANDIDATE_INVERSION_TERMINAL}")"
    CANDIDATE_INVERSION_RECORDS_DIRECTORY="$(canonical_direct_directory "candidate inversion records directory" "${CANDIDATE_INVERSION_RECORDS_DIRECTORY}")"
    case "${CANDIDATE_INVERSION_TERMINAL}" in
      "${CANDIDATE_EVIDENCE_ROOT}"/*) ;;
      *) echo "error: candidate inversion terminal must be inside the candidate evidence root" >&2; exit 2 ;;
    esac
    case "${CANDIDATE_INVERSION_RECORDS_DIRECTORY}" in
      "${CANDIDATE_EVIDENCE_ROOT}"/*) ;;
      *) echo "error: candidate inversion records directory must be inside the candidate evidence root" >&2; exit 2 ;;
    esac
    if [[ "$(/usr/bin/sha256sum -- "${CANDIDATE_INVERSION_TERMINAL}" | /usr/bin/awk '{print $1}')" != "${CANDIDATE_INVERSION_TERMINAL_SHA256}" ]]; then
      echo "error: candidate inversion terminal SHA-256 differs" >&2
      exit 2
    fi
    if [[ ! -f "${CANDIDATE_ATTESTATION}" || -L "${CANDIDATE_ATTESTATION}" ]]; then
      echo "error: candidate attestation is not a regular non-symlink file" >&2
      exit 2
    fi
    CANDIDATE_ATTESTATION="$(/usr/bin/realpath -- "${CANDIDATE_ATTESTATION}")"
    if [[ ! "${SOURCE_TRANSITION_POLICY_SHA256}" =~ ^[0-9a-f]{64}$ || ! -f "${SOURCE_TRANSITION_POLICY}" || -L "${SOURCE_TRANSITION_POLICY}" ]]; then
      echo "error: source-transition policy or exact SHA-256 is invalid" >&2
      exit 2
    fi
    SOURCE_TRANSITION_POLICY="$(/usr/bin/realpath -- "${SOURCE_TRANSITION_POLICY}")"
    if [[ ! "${SOURCE_TRANSITION_VERIFIER_SHA256}" =~ ^[0-9a-f]{64}$ || ! -f "${SOURCE_TRANSITION_VERIFIER}" || -L "${SOURCE_TRANSITION_VERIFIER}" ]]; then
      echo "error: source-transition verifier or exact SHA-256 is invalid" >&2
      exit 2
    fi
    SOURCE_TRANSITION_VERIFIER="$(/usr/bin/realpath -- "${SOURCE_TRANSITION_VERIFIER}")"
    if [[ "$(/usr/bin/sha256sum -- "${SOURCE_TRANSITION_POLICY}" | /usr/bin/awk '{print $1}')" != "${SOURCE_TRANSITION_POLICY_SHA256}" ]]; then
      echo "error: source-transition policy SHA-256 differs" >&2
      exit 2
    fi
    if [[ "$(/usr/bin/sha256sum -- "${SOURCE_TRANSITION_VERIFIER}" | /usr/bin/awk '{print $1}')" != "${SOURCE_TRANSITION_VERIFIER_SHA256}" ]]; then
      echo "error: source-transition verifier SHA-256 differs" >&2
      exit 2
    fi
    case "${SOURCE_TRANSITION_POLICY}" in
      "${REPO_ROOT}"/*) echo "error: source-transition policy must be outside the candidate repository" >&2; exit 2 ;;
    esac
    case "${SOURCE_TRANSITION_VERIFIER}" in
      "${REPO_ROOT}"/*) echo "error: source-transition verifier must be outside the candidate repository" >&2; exit 2 ;;
    esac
    if [[ ! -f "${M4_CANDIDATE_MANIFEST}" || -L "${M4_CANDIDATE_MANIFEST}" ]]; then
      echo "error: M4 candidate manifest is not a regular non-symlink file" >&2
      exit 2
    fi
    M4_CANDIDATE_MANIFEST="$(/usr/bin/realpath -- "${M4_CANDIDATE_MANIFEST}")"
    if [[ "${M4_CANDIDATE_MANIFEST}" != "${CANDIDATE_EVIDENCE_ROOT}/"* ]]; then
      echo "error: M4 candidate manifest must be inside the candidate evidence root" >&2
      exit 2
    fi
    if [[ -n "$(/usr/bin/git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]]; then
      echo "error: package source tree must be clean and committed" >&2
      exit 1
    fi
    SOURCE_COMMIT="$(/usr/bin/git -C "${REPO_ROOT}" rev-parse HEAD)"
    if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
      echo "error: package source identity is invalid" >&2
      exit 1
    fi
    M4_RELEASE_SOURCE_COMMIT="$(/usr/bin/jq -er '.release_source_commit' "${M4_CANDIDATE_MANIFEST}")"
    M4_SOURCE_AUDIT_RELATIVE="$(/usr/bin/jq -er '.evidence.m4_source_audit.bundle_relative_path' "${M4_CANDIDATE_MANIFEST}")"
    M4_SOURCE_AUDIT_SHA256="$(/usr/bin/jq -er '.evidence.m4_source_audit.sha256' "${M4_CANDIDATE_MANIFEST}")"
    if [[ "${M4_RELEASE_SOURCE_COMMIT}" != "${SOURCE_COMMIT}" || ! "${M4_SOURCE_AUDIT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "error: M4 source preflight identity differs" >&2
      exit 2
    fi
    if ! /usr/bin/jq -e '
      .evidence.m4_source_audit.role == "m4_source_audit" and
      .evidence.m4_source_audit.schema == "gpmeep-v110-release-source-transition-v3" and
      (.evidence.m4_source_audit.content_id | test("^[0-9a-f]{64}$")) and
      (.evidence.m4_source_audit.size_bytes | type == "number" and . > 0)
    ' "${M4_CANDIDATE_MANIFEST}" >/dev/null; then
      echo "error: M4 source preflight reference differs" >&2
      exit 2
    fi
    case "/${M4_SOURCE_AUDIT_RELATIVE}/" in
      *"/../"*|*"/./"*|"//"*|*"//"*) echo "error: M4 source preflight path is unsafe" >&2; exit 2 ;;
    esac
    M4_SOURCE_AUDIT="$(/usr/bin/realpath -- "${CANDIDATE_EVIDENCE_ROOT}/${M4_SOURCE_AUDIT_RELATIVE}")"
    if [[ "${M4_SOURCE_AUDIT}" != "${CANDIDATE_EVIDENCE_ROOT}/"* || ! -f "${M4_SOURCE_AUDIT}" || -L "${M4_SOURCE_AUDIT}" ]]; then
      echo "error: M4 source preflight report is unsafe" >&2
      exit 2
    fi
    if [[ "$(/usr/bin/sha256sum -- "${M4_SOURCE_AUDIT}" | /usr/bin/awk '{print $1}')" != "${M4_SOURCE_AUDIT_SHA256}" ]]; then
      echo "error: M4 source preflight report SHA-256 differs" >&2
      exit 2
    fi
    if ! (
      cd /
      /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
        /usr/bin/python3 -I -S "${SOURCE_TRANSITION_VERIFIER}" \
        --verify "${M4_SOURCE_AUDIT}" --repo "${REPO_ROOT}" \
        --authority-policy "${SOURCE_TRANSITION_POLICY}" \
        --authority-policy-sha256 "${SOURCE_TRANSITION_POLICY_SHA256}" >/dev/null
    ); then
      echo "error: independently pinned source-transition preflight failed" >&2
      exit 2
    fi
  else
    if [[ -n "${CANDIDATE_EVIDENCE_ROOT}" || -n "${CANDIDATE_ATTESTATION}" || -n "${CANDIDATE_ATTESTATION_SHA256}" || -n "${M4_CANDIDATE_MANIFEST}" || -n "${SOURCE_TRANSITION_POLICY}" || -n "${SOURCE_TRANSITION_POLICY_SHA256}" || -n "${SOURCE_TRANSITION_VERIFIER}" || -n "${SOURCE_TRANSITION_VERIFIER_SHA256}" || -n "${CANDIDATE_INVERSION_TERMINAL}" || -n "${CANDIDATE_INVERSION_TERMINAL_SHA256}" || -n "${CANDIDATE_INVERSION_TERMINAL_ID}" || -n "${CANDIDATE_INVERSION_RECORDS_DIRECTORY}" ]]; then
      echo "error: development build cannot accept candidate authority arguments" >&2
      exit 2
    fi
    if [[ -n "$(/usr/bin/git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]]; then
      echo "error: package source tree must be clean and committed" >&2
      exit 1
    fi
    SOURCE_COMMIT="$(/usr/bin/git -C "${REPO_ROOT}" rev-parse HEAD)"
    if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
      echo "error: package source identity is invalid" >&2
      exit 1
    fi
  fi
  /usr/bin/python3 -I "${V110_CONTRACT_VALIDATOR}" --contract-only
fi
if [[ -z "${SOURCE_COMMIT:-}" ]]; then
  if [[ -n "$(/usr/bin/git -C "${REPO_ROOT}" status --porcelain --untracked-files=all)" ]]; then
    echo "error: package source tree must be clean and committed" >&2
    exit 1
  fi
  SOURCE_COMMIT="$(/usr/bin/git -C "${REPO_ROOT}" rev-parse HEAD)"
  [[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "error: package source identity is invalid" >&2; exit 1; }
fi

prepare_v110_gate_runtime() {
  V110_GATE_PYTHON="${PREFIX}/bin/python3.11"
  if [[ -f "${REPO_ROOT}/packaging/v110-development-contract.json" \
        && ! -x "${V110_GATE_PYTHON}" ]]; then
    echo "error: exact package builder environment lacks its Python gate runtime" >&2
    return 1
  fi
  if [[ "${BUILD_MODE}" == "candidate" ]]; then
    # Candidate evidence replays full numerical results before conda-build
    # creates its recipe environments. Use the already contracted analysis
    # runtime; the packaging environment remains the exact builder lock.
    /bin/bash -p "${SCRIPT_DIR}/create-env.sh" v110-test
    V110_GATE_PYTHON="$(canonical_direct_file "candidate gate Python" \
      "${REPO_ROOT}/.envs/gpmeep-v1.1-test/bin/python3.11")"
    local python_sha256
    python_sha256="$(/usr/bin/sha256sum -- "${V110_GATE_PYTHON}" | /usr/bin/awk '{print $1}')"
    "${V110_GATE_PYTHON}" -I -S "${V110_RUNTIME_VERIFIER}" \
      --repo "${REPO_ROOT}" --source-commit "${SOURCE_COMMIT}" \
      --python-sha256 "${python_sha256}" >/dev/null
  fi
}

/bin/bash -p "${SCRIPT_DIR}/create-env.sh" "${ENVIRONMENT_PROFILE}"
if [[ ! -x "${MAMBA}" || ! -x "${PREFIX}/bin/conda" ]]; then
  echo "error: exact package builder environment is incomplete" >&2
  exit 1
fi
prepare_v110_gate_runtime
if [[ -e "${OUTPUT_DIRECTORY}" ]]; then
  if [[ ! -d "${OUTPUT_DIRECTORY}" || -L "${OUTPUT_DIRECTORY}" ]]; then
    echo "error: output path is not a safe directory" >&2
    exit 1
  fi
else
  /usr/bin/install -d -m 0755 "${OUTPUT_DIRECTORY}"
fi
GPMEEP_V110_CANDIDATE_RECEIPT=""
if [[ -f "${REPO_ROOT}/packaging/v110-development-contract.json" ]]; then
  V110_GATE_DIRECTORY="${OUTPUT_DIRECTORY}/v110-gates-${SOURCE_COMMIT}"
  if [[ -e "${V110_GATE_DIRECTORY}" || -L "${V110_GATE_DIRECTORY}" ]]; then
    echo "error: v1.1.0 gate output already exists: ${V110_GATE_DIRECTORY}" >&2
    exit 1
  fi
  /usr/bin/install -d -m 0700 "${V110_GATE_DIRECTORY}"
  V110_CONTRACT_GATE="${V110_GATE_DIRECTORY}/contract.json"
  V110_REGISTRY_GATE="${V110_GATE_DIRECTORY}/registry.json"
  V110_ATTESTATION_GATE="${V110_GATE_DIRECTORY}/attestation.json"
  GPMEEP_V110_CANDIDATE_RECEIPT="${V110_GATE_DIRECTORY}/candidate-build-receipt.json"
  /usr/bin/install -m 0600 /dev/null "${V110_CONTRACT_GATE}"
  /usr/bin/install -m 0600 /dev/null "${V110_REGISTRY_GATE}"
  if [[ "${BUILD_MODE}" == "candidate" ]]; then
    # Preserve full external-input validation; the receipt replays the source-only gate.
    V110_FULL_CONTRACT_GATE="${V110_GATE_DIRECTORY}/contract-full-inputs.json"
    /usr/bin/install -m 0600 /dev/null "${V110_FULL_CONTRACT_GATE}"
    "${V110_GATE_PYTHON}" -I "${V110_CONTRACT_VALIDATOR}" \
      --workspace-root "${REPO_ROOT}/.." --json >"${V110_FULL_CONTRACT_GATE}"
    "${V110_GATE_PYTHON}" -I "${V110_CONTRACT_VALIDATOR}" --contract-only --json \
      >"${V110_CONTRACT_GATE}"
    CLOSURE_RESULT="$("${V110_GATE_PYTHON}" -I "${V110_REGISTRY_CLOSER}" \
      --verify --terminal "${CANDIDATE_INVERSION_TERMINAL}" \
      --evidence-root "${CANDIDATE_EVIDENCE_ROOT}")"
    if ! /usr/bin/jq -e --arg terminal_id "${CANDIDATE_INVERSION_TERMINAL_ID}" '
      keys == ["schema", "state", "terminal_id"] and
      .schema == "gpmeep-v110-closed-inversion-registry-v1" and
      .state == "PASS" and .terminal_id == $terminal_id
    ' <<<"${CLOSURE_RESULT}" >/dev/null; then
      echo "error: candidate inversion terminal ID or replay result differs" >&2
      exit 2
    fi
    TERMINAL_RECORDS_RELATIVE="$(/usr/bin/jq -er '.records_directory | select(type == "string")' "${CANDIDATE_INVERSION_TERMINAL}")"
    TERMINAL_RECORDS_DIRECTORY="$(canonical_direct_directory "terminal-derived inversion records directory" "${CANDIDATE_EVIDENCE_ROOT}/${TERMINAL_RECORDS_RELATIVE}")"
    if [[ "${TERMINAL_RECORDS_DIRECTORY}" != "${CANDIDATE_INVERSION_RECORDS_DIRECTORY}" ]]; then
      echo "error: explicit inversion records directory differs from the closed terminal" >&2
      exit 2
    fi
    if [[ "$(/usr/bin/sha256sum -- "${CANDIDATE_INVERSION_TERMINAL}" | /usr/bin/awk '{print $1}')" != "${CANDIDATE_INVERSION_TERMINAL_SHA256}" ]]; then
      echo "error: candidate inversion terminal changed during replay" >&2
      exit 2
    fi
    "${V110_GATE_PYTHON}" -I "${V110_REGISTRY_VALIDATOR}" \
      --require-closed --evidence-root "${CANDIDATE_EVIDENCE_ROOT}" \
      --records-directory "${CANDIDATE_INVERSION_RECORDS_DIRECTORY}" --json \
      >"${V110_REGISTRY_GATE}"
    "${V110_GATE_PYTHON}" -I "${V110_ATTESTER}" \
      --verify "${CANDIDATE_ATTESTATION}" \
      --expected-sha256 "${CANDIDATE_ATTESTATION_SHA256}" --json \
      >/dev/null
    /usr/bin/install -m 0600 "${CANDIDATE_ATTESTATION}" "${V110_ATTESTATION_GATE}"
    if [[ "$(/usr/bin/sha256sum -- "${V110_ATTESTATION_GATE}" | /usr/bin/awk '{print $1}')" != "${CANDIDATE_ATTESTATION_SHA256}" ]] || ! /usr/bin/cmp -s -- "${CANDIDATE_ATTESTATION}" "${V110_ATTESTATION_GATE}"; then
      echo "error: retained M0 attestation gate differs from the verified original" >&2
      exit 2
    fi
    "${V110_GATE_PYTHON}" -I "${V110_RECEIPT_TOOL}" --create --mode candidate \
      --source-commit "${SOURCE_COMMIT}" \
      --contract-gate "${V110_CONTRACT_GATE}" \
      --registry-gate "${V110_REGISTRY_GATE}" \
      --attestation-gate "${V110_ATTESTATION_GATE}" \
      --expected-attestation-sha256 "${CANDIDATE_ATTESTATION_SHA256}" \
      --m4-candidate-manifest "${M4_CANDIDATE_MANIFEST}" \
      --source-transition-policy "${SOURCE_TRANSITION_POLICY}" \
      --expected-source-transition-policy-sha256 "${SOURCE_TRANSITION_POLICY_SHA256}" \
      --source-transition-verifier "${SOURCE_TRANSITION_VERIFIER}" \
      --expected-source-transition-verifier-sha256 "${SOURCE_TRANSITION_VERIFIER_SHA256}" \
      --closed-registry-terminal "${CANDIDATE_INVERSION_TERMINAL}" \
      --expected-closed-registry-sha256 "${CANDIDATE_INVERSION_TERMINAL_SHA256}" \
      --expected-closed-registry-terminal-id "${CANDIDATE_INVERSION_TERMINAL_ID}" \
      --closed-registry-records-directory "${CANDIDATE_INVERSION_RECORDS_DIRECTORY}" \
      --evidence-root "${CANDIDATE_EVIDENCE_ROOT}" \
      --output "${GPMEEP_V110_CANDIDATE_RECEIPT}"
  else
    "${V110_GATE_PYTHON}" -I "${V110_CONTRACT_VALIDATOR}" --contract-only --json \
      >"${V110_CONTRACT_GATE}"
    "${V110_GATE_PYTHON}" -I "${V110_REGISTRY_VALIDATOR}" --allow-open --json \
      >"${V110_REGISTRY_GATE}"
    "${V110_GATE_PYTHON}" -I "${V110_RECEIPT_TOOL}" --create --mode development \
      --source-commit "${SOURCE_COMMIT}" \
      --contract-gate "${V110_CONTRACT_GATE}" \
      --registry-gate "${V110_REGISTRY_GATE}" \
      --output "${GPMEEP_V110_CANDIDATE_RECEIPT}"
  fi
fi
if /usr/bin/find "${OUTPUT_DIRECTORY}" -mindepth 1 -maxdepth 2 \
    \( -name "gpmeep-${DISTRIBUTION_VERSION}-*.conda" \
       -o -name "gpmeep-${DISTRIBUTION_VERSION}-*.tar.bz2" \) \
    -print -quit | /usr/bin/grep -q .; then
  echo "error: output directory already contains a gpmeep ${DISTRIBUTION_VERSION} package" >&2
  echo "hint: choose a clean build directory with --output-dir /absolute/path" >&2
  exit 1
fi

# Conda's path source copies ignored files as well as Git-tracked content.
# Create and retain a checksum-bound archive of the exact commit, extract it
# into a fresh tree, and give only that tree to the recipe.
SOURCE_ARCHIVE="${OUTPUT_DIRECTORY}/gpmeep-source-${SOURCE_COMMIT}.tar.gz"
SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE}.sha256"
SOURCE_TREE="${OUTPUT_DIRECTORY}/source-${SOURCE_COMMIT}"
for path in "${SOURCE_ARCHIVE}" "${SOURCE_ARCHIVE_SHA256}" "${SOURCE_TREE}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "error: source staging path already exists: ${path}" >&2
    exit 1
  fi
done
/usr/bin/install -d -m 0755 "${SOURCE_TREE}"
/usr/bin/git -C "${REPO_ROOT}" archive --format=tar.gz \
  --prefix=gpmeep-source/ --output="${SOURCE_ARCHIVE}" "${SOURCE_COMMIT}"
/usr/bin/tar -xzf "${SOURCE_ARCHIVE}" --strip-components=1 -C "${SOURCE_TREE}"
SOURCE_ARCHIVE_DIGEST="$(/usr/bin/sha256sum "${SOURCE_ARCHIVE}" | /usr/bin/awk '{print $1}')"
/usr/bin/printf '%s  %s\n' "${SOURCE_ARCHIVE_DIGEST}" \
  "$(/usr/bin/basename "${SOURCE_ARCHIVE}")" >"${SOURCE_ARCHIVE_SHA256}"
if /usr/bin/find "${SOURCE_TREE}" -mindepth 1 \
    \( -name .git -o -name .envs -o -name benchmark-results \) \
    -print -quit | /usr/bin/grep -q .; then
  echo "error: exact source snapshot contains a forbidden local-state path" >&2
  exit 1
fi

export GPMEEP_SOURCE_COMMIT="${SOURCE_COMMIT}"
export GPMEEP_CUDA_ARCHS="${CUDA_ARCHITECTURES}"
export GPMEEP_SOURCE_TREE="${SOURCE_TREE}"
export GPMEEP_V110_CANDIDATE_RECEIPT
export GPMEEP_V110_BUILD_MODE="${BUILD_MODE}"
export CONDA_BLD_PATH="${OUTPUT_DIRECTORY}"
export PYTHONNOUSERSITE=1
/usr/bin/install -d -m 0700 "${MAMBA_PROCESS_CACHE}"
if /usr/bin/env XDG_CACHE_HOME="${MAMBA_PROCESS_CACHE}" \
    "${MAMBA}" --no-rc run --clean-env \
    --env "GPMEEP_SOURCE_COMMIT=${GPMEEP_SOURCE_COMMIT}" \
    --env "GPMEEP_CUDA_ARCHS=${GPMEEP_CUDA_ARCHS}" \
    --env "GPMEEP_SOURCE_TREE=${GPMEEP_SOURCE_TREE}" \
    --env "GPMEEP_V110_CANDIDATE_RECEIPT=${GPMEEP_V110_CANDIDATE_RECEIPT}" \
    --env "GPMEEP_V110_BUILD_MODE=${GPMEEP_V110_BUILD_MODE}" \
    --env "CONDA_BLD_PATH=${CONDA_BLD_PATH}" \
    --env "CPU_COUNT=${BUILD_JOBS}" \
    --env "XDG_CACHE_HOME=${MAMBA_PROCESS_CACHE}" \
    --env PYTHONNOUSERSITE=1 \
    --root-prefix "${MAMBA_ROOT}" --prefix "${PREFIX}" \
    conda build "${REPO_ROOT}/packaging/conda" \
      --override-channels --channel conda-forge --no-anaconda-upload; then
  :
else
  CONDA_BUILD_RC=$?
  echo "error: conda build failed with status ${CONDA_BUILD_RC}" >&2
  exit "${CONDA_BUILD_RC}"
fi

if [[ -d "${OUTPUT_DIRECTORY}/broken" ]] && \
   /usr/bin/find "${OUTPUT_DIRECTORY}/broken" -type f \
     \( -name "gpmeep-${DISTRIBUTION_VERSION}-*.conda" \
        -o -name "gpmeep-${DISTRIBUTION_VERSION}-*.tar.bz2" \) \
     -print -quit | /usr/bin/grep -q .; then
  echo "error: conda placed a gpmeep ${DISTRIBUTION_VERSION} artifact in broken/" >&2
  exit 1
fi

mapfile -t PACKAGES < <(
  /usr/bin/find "${OUTPUT_DIRECTORY}/linux-64" -maxdepth 1 -type f \
    \( -name "gpmeep-${DISTRIBUTION_VERSION}-*.conda" \
       -o -name "gpmeep-${DISTRIBUTION_VERSION}-*.tar.bz2" \) \
    -print
)
if [[ "${#PACKAGES[@]}" -ne 1 ]]; then
  echo "error: package build produced ${#PACKAGES[@]} gpmeep artifacts" >&2
  exit 1
fi
PACKAGE="${PACKAGES[0]}"
PACKAGE_BASENAME="$(/usr/bin/basename "${PACKAGE}")"
if [[ "${BUILD_MODE}" == "development" ]]; then
  case "${PACKAGE_BASENAME}" in
    gpmeep-${DISTRIBUTION_VERSION}-*_development.conda|gpmeep-${DISTRIBUTION_VERSION}-*_development.tar.bz2) ;;
    *) echo "error: development artifact omits its _development build identity" >&2; exit 1 ;;
  esac
else
  case "${PACKAGE_BASENAME}" in
    *_development.conda|*_development.tar.bz2)
      echo "error: candidate artifact carries a development build identity" >&2
      exit 1
      ;;
  esac
fi
PACKAGE_DIGEST="$(/usr/bin/sha256sum "${PACKAGE}" | /usr/bin/awk '{print $1}')"
PACKAGE_SIZE="$('/usr/bin/stat' -c %s -- "${PACKAGE}")"
NONPUBLISHABLE_MARKER=""
if [[ "${BUILD_MODE}" == "development" ]]; then
  NONPUBLISHABLE_MARKER="${PACKAGE}.NON_PUBLISHABLE"
  if [[ -e "${NONPUBLISHABLE_MARKER}" || -L "${NONPUBLISHABLE_MARKER}" ]]; then
    echo "error: non-publishable marker path already exists" >&2
    exit 1
  fi
  if ! (
    set -o noclobber
    umask 077
    /usr/bin/printf '%s\n' \
      "{\"package\":{\"file_name\":\"${PACKAGE_BASENAME}\",\"sha256\":\"${PACKAGE_DIGEST}\",\"size_bytes\":${PACKAGE_SIZE}},\"schema\":\"gpmeep-non-publishable-development-artifact-v1\",\"source_commit\":\"${SOURCE_COMMIT}\",\"state\":\"NON_PUBLISHABLE\"}" \
      >"${NONPUBLISHABLE_MARKER}"
  ); then
    echo "error: cannot create the exclusive non-publishable marker" >&2
    exit 1
  fi
  /usr/bin/chmod 0400 "${NONPUBLISHABLE_MARKER}"
fi
/usr/bin/printf '%s  %s\n' "${PACKAGE_DIGEST}" \
  "${PACKAGE_BASENAME}" >"${PACKAGE}.sha256"
printf 'gpmeep-package=%s\n' "${PACKAGE}"
if [[ -n "${NONPUBLISHABLE_MARKER}" ]]; then
  printf 'gpmeep-non-publishable-marker=%s\n' "${NONPUBLISHABLE_MARKER}"
fi
printf 'source-commit=%s\n' "${SOURCE_COMMIT}"
CHECKSUM_PATHS=("${PACKAGE}" "${PACKAGE}.sha256" "${SOURCE_ARCHIVE}" "${SOURCE_ARCHIVE_SHA256}")
if [[ -n "${NONPUBLISHABLE_MARKER}" ]]; then
  CHECKSUM_PATHS+=("${NONPUBLISHABLE_MARKER}")
fi
/usr/bin/sha256sum "${CHECKSUM_PATHS[@]}"
