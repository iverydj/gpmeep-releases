#!/bin/bash -p
set -Eeuo pipefail
set +x
set +v
umask 077

[[ $- == *p* ]] || { printf '%s\n' 'error: protected Bash mode is required' >&2; exit 2; }

REPO=
PACKAGE=
FINAL_RECEIPT=
CANDIDATE_RECEIPT=
QUALIFICATION_MANIFEST=
EVIDENCE_ROOT=
CANDIDATE_CONTRACT_GATE=
CANDIDATE_REGISTRY_GATE=
CANDIDATE_ATTESTATION=
M4_CANDIDATE_MANIFEST=
CANDIDATE_EVIDENCE_ROOT=
SOURCE_TRANSITION_POLICY=
SOURCE_TRANSITION_POLICY_SHA256=
SOURCE_TRANSITION_VERIFIER=
SOURCE_TRANSITION_VERIFIER_SHA256=
FINAL_AUDIT_AUTHORITY=
FINAL_AUDIT_AUTHORITY_SHA256=
ASSET_DIRECTORY=
REPOSITORY=
EXECUTE=false
RESUME=false

usage() {
  printf '%s\n' "usage: $0 --repo PATH --package FILE --final-receipt FILE --candidate-receipt FILE --qualification-manifest FILE --evidence-root PATH --candidate-contract-gate FILE --candidate-registry-gate FILE --candidate-attestation FILE --m4-candidate-manifest FILE --candidate-evidence-root PATH --source-transition-policy FILE --source-transition-policy-sha256 HEX --source-transition-verifier FILE --source-transition-verifier-sha256 HEX --final-audit-authority FILE --final-audit-authority-sha256 HEX --asset-directory PATH --repository OWNER/REPO [--execute [--resume]]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo|--package|--final-receipt|--candidate-receipt|--qualification-manifest|--evidence-root|--candidate-contract-gate|--candidate-registry-gate|--candidate-attestation|--m4-candidate-manifest|--candidate-evidence-root|--source-transition-policy|--source-transition-policy-sha256|--source-transition-verifier|--source-transition-verifier-sha256|--final-audit-authority|--final-audit-authority-sha256|--asset-directory|--repository)
      [[ $# -ge 2 && -n "$2" ]] || { usage; exit 2; }
      case "$1" in
        --repo) REPO=$2 ;;
        --package) PACKAGE=$2 ;;
        --final-receipt) FINAL_RECEIPT=$2 ;;
        --candidate-receipt) CANDIDATE_RECEIPT=$2 ;;
        --qualification-manifest) QUALIFICATION_MANIFEST=$2 ;;
        --evidence-root) EVIDENCE_ROOT=$2 ;;
        --candidate-contract-gate) CANDIDATE_CONTRACT_GATE=$2 ;;
        --candidate-registry-gate) CANDIDATE_REGISTRY_GATE=$2 ;;
        --candidate-attestation) CANDIDATE_ATTESTATION=$2 ;;
        --m4-candidate-manifest) M4_CANDIDATE_MANIFEST=$2 ;;
        --candidate-evidence-root) CANDIDATE_EVIDENCE_ROOT=$2 ;;
        --source-transition-policy) SOURCE_TRANSITION_POLICY=$2 ;;
        --source-transition-policy-sha256) SOURCE_TRANSITION_POLICY_SHA256=$2 ;;
        --source-transition-verifier) SOURCE_TRANSITION_VERIFIER=$2 ;;
        --source-transition-verifier-sha256) SOURCE_TRANSITION_VERIFIER_SHA256=$2 ;;
        --final-audit-authority) FINAL_AUDIT_AUTHORITY=$2 ;;
        --final-audit-authority-sha256) FINAL_AUDIT_AUTHORITY_SHA256=$2 ;;
        --asset-directory) ASSET_DIRECTORY=$2 ;;
        --repository) REPOSITORY=$2 ;;
      esac
      shift 2
      ;;
    --execute) EXECUTE=true; shift ;;
    --resume) RESUME=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

for value in "$REPO" "$PACKAGE" "$FINAL_RECEIPT" "$CANDIDATE_RECEIPT" "$QUALIFICATION_MANIFEST" "$EVIDENCE_ROOT" "$CANDIDATE_CONTRACT_GATE" "$CANDIDATE_REGISTRY_GATE" "$CANDIDATE_ATTESTATION" "$M4_CANDIDATE_MANIFEST" "$CANDIDATE_EVIDENCE_ROOT" "$SOURCE_TRANSITION_POLICY" "$SOURCE_TRANSITION_POLICY_SHA256" "$SOURCE_TRANSITION_VERIFIER" "$SOURCE_TRANSITION_VERIFIER_SHA256" "$FINAL_AUDIT_AUTHORITY" "$FINAL_AUDIT_AUTHORITY_SHA256" "$ASSET_DIRECTORY" "$REPOSITORY"; do
  [[ -n "$value" ]] || { usage; exit 2; }
done
[[ "$RESUME" != true || "$EXECUTE" == true ]] || { printf '%s\n' 'error: --resume requires --execute' >&2; exit 2; }
[[ "$REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { printf '%s\n' 'error: invalid GitHub repository name' >&2; exit 2; }
[[ "$REPOSITORY" == iverydj/gpmeep ]] || { printf '%s\n' 'error: source publication is restricted to iverydj/gpmeep; use publish-v110-public.py for the reviewed public snapshot' >&2; exit 2; }

canonical_nonsymlink_file() {
  local lexical resolved
  lexical=$(/usr/bin/realpath --canonicalize-missing --no-symlinks -- "$1")
  [[ -f "$lexical" && ! -L "$lexical" ]] || { printf 'error: unsafe release input: %s\n' "$1" >&2; exit 2; }
  resolved=$(/usr/bin/realpath -- "$lexical")
  [[ "$resolved" == "$lexical" ]] || { printf 'error: release input path contains a symlink: %s\n' "$1" >&2; exit 2; }
  printf '%s' "$resolved"
}

canonical_nonsymlink_directory() {
  local lexical resolved
  lexical=$(/usr/bin/realpath --canonicalize-missing --no-symlinks -- "$1")
  [[ -d "$lexical" && ! -L "$lexical" ]] || { printf 'error: unsafe release directory: %s\n' "$1" >&2; exit 2; }
  resolved=$(/usr/bin/realpath -- "$lexical")
  [[ "$resolved" == "$lexical" ]] || { printf 'error: release directory path contains a symlink: %s\n' "$1" >&2; exit 2; }
  printf '%s' "$resolved"
}

REPO=$(canonical_nonsymlink_directory "$REPO")
PACKAGE=$(canonical_nonsymlink_file "$PACKAGE")
FINAL_RECEIPT=$(canonical_nonsymlink_file "$FINAL_RECEIPT")
CANDIDATE_RECEIPT=$(canonical_nonsymlink_file "$CANDIDATE_RECEIPT")
QUALIFICATION_MANIFEST=$(canonical_nonsymlink_file "$QUALIFICATION_MANIFEST")
EVIDENCE_ROOT=$(canonical_nonsymlink_directory "$EVIDENCE_ROOT")
CANDIDATE_CONTRACT_GATE=$(canonical_nonsymlink_file "$CANDIDATE_CONTRACT_GATE")
CANDIDATE_REGISTRY_GATE=$(canonical_nonsymlink_file "$CANDIDATE_REGISTRY_GATE")
CANDIDATE_ATTESTATION=$(canonical_nonsymlink_file "$CANDIDATE_ATTESTATION")
M4_CANDIDATE_MANIFEST=$(canonical_nonsymlink_file "$M4_CANDIDATE_MANIFEST")
CANDIDATE_EVIDENCE_ROOT=$(canonical_nonsymlink_directory "$CANDIDATE_EVIDENCE_ROOT")
SOURCE_TRANSITION_POLICY=$(canonical_nonsymlink_file "$SOURCE_TRANSITION_POLICY")
[[ "$SOURCE_TRANSITION_POLICY_SHA256" =~ ^[0-9a-f]{64}$ ]] || { printf '%s\n' 'error: invalid source-transition policy SHA-256' >&2; exit 2; }
[[ "$(/usr/bin/sha256sum -- "$SOURCE_TRANSITION_POLICY" | /usr/bin/awk '{print $1}')" == "$SOURCE_TRANSITION_POLICY_SHA256" ]] || { printf '%s\n' 'error: source-transition policy SHA-256 differs' >&2; exit 2; }
SOURCE_TRANSITION_VERIFIER=$(canonical_nonsymlink_file "$SOURCE_TRANSITION_VERIFIER")
[[ "$SOURCE_TRANSITION_VERIFIER_SHA256" =~ ^[0-9a-f]{64}$ ]] || { printf '%s\n' 'error: invalid source-transition verifier SHA-256' >&2; exit 2; }
[[ "$(/usr/bin/sha256sum -- "$SOURCE_TRANSITION_VERIFIER" | /usr/bin/awk '{print $1}')" == "$SOURCE_TRANSITION_VERIFIER_SHA256" ]] || { printf '%s\n' 'error: source-transition verifier SHA-256 differs' >&2; exit 2; }
case "$SOURCE_TRANSITION_POLICY" in
  "$REPO"/*) printf '%s\n' 'error: source-transition policy must be external to the repository' >&2; exit 2 ;;
esac
case "$SOURCE_TRANSITION_VERIFIER" in
  "$REPO"/*) printf '%s\n' 'error: source-transition verifier must be external to the repository' >&2; exit 2 ;;
esac
FINAL_AUDIT_AUTHORITY=$(canonical_nonsymlink_file "$FINAL_AUDIT_AUTHORITY")
[[ "$FINAL_AUDIT_AUTHORITY_SHA256" =~ ^[0-9a-f]{64}$ ]] || { printf '%s\n' 'error: invalid final audit authority SHA-256' >&2; exit 2; }
[[ "$(/usr/bin/sha256sum -- "$FINAL_AUDIT_AUTHORITY" | /usr/bin/awk '{print $1}')" == "$FINAL_AUDIT_AUTHORITY_SHA256" ]] || { printf '%s\n' 'error: final audit authority SHA-256 differs' >&2; exit 2; }
ASSET_DIRECTORY=$(canonical_nonsymlink_directory "$ASSET_DIRECTORY")
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || { printf '%s\n' 'error: repository is not a Git worktree' >&2; exit 2; }
[[ "$M4_CANDIDATE_MANIFEST" == "$CANDIDATE_EVIDENCE_ROOT/"* ]] || { printf '%s\n' 'error: M4 candidate manifest must be inside candidate evidence root' >&2; exit 2; }
case "$ASSET_DIRECTORY/" in
  "$REPO/"*|"$EVIDENCE_ROOT/"*|"$CANDIDATE_EVIDENCE_ROOT/"*)
    printf '%s\n' 'error: asset directory must be outside repository and evidence roots' >&2
    exit 2
    ;;
esac

FINAL_TOOL="$REPO/scripts/v110-final-release-receipt.py"
CANDIDATE_TOOL="$REPO/scripts/v110-release-gate-receipt.py"
ASSET_TOOL="$REPO/scripts/v110-release-assets.py"
CONTRACT="$REPO/packaging/v110-development-contract.json"
INVERSION_SCHEMA="$REPO/packaging/v110-performance-inversion-record.schema.json"
INVENTORY="$REPO/packaging/v110-performance-inversion-inventory.json"
for file in "$FINAL_TOOL" "$CANDIDATE_TOOL" "$ASSET_TOOL" "$CONTRACT" "$INVERSION_SCHEMA" "$INVENTORY"; do
  [[ -f "$file" && ! -L "$file" ]] || { printf 'error: missing release verifier input: %s\n' "$file" >&2; exit 2; }
done

SOURCE_PREFLIGHT_COMMIT=$(/usr/bin/jq -er '.release_source_commit' "$M4_CANDIDATE_MANIFEST")
[[ "$SOURCE_PREFLIGHT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { printf '%s\n' 'error: source preflight commit is invalid' >&2; exit 2; }
SOURCE_PREFLIGHT_RELATIVE=$(/usr/bin/jq -er '.evidence.m4_source_audit.bundle_relative_path' "$M4_CANDIDATE_MANIFEST")
SOURCE_PREFLIGHT_SHA256=$(/usr/bin/jq -er '.evidence.m4_source_audit.sha256' "$M4_CANDIDATE_MANIFEST")
[[ "$SOURCE_PREFLIGHT_SHA256" =~ ^[0-9a-f]{64}$ ]] || { printf '%s\n' 'error: source preflight report SHA-256 is invalid' >&2; exit 2; }
/usr/bin/jq -e '
  .evidence.m4_source_audit.role == "m4_source_audit" and
  .evidence.m4_source_audit.schema == "gpmeep-v110-release-source-transition-v3" and
  (.evidence.m4_source_audit.content_id | test("^[0-9a-f]{64}$")) and
  (.evidence.m4_source_audit.size_bytes | type == "number" and . > 0)
' "$M4_CANDIDATE_MANIFEST" >/dev/null || { printf '%s\n' 'error: source preflight manifest reference differs' >&2; exit 2; }
case "/$SOURCE_PREFLIGHT_RELATIVE/" in
  *"/../"*|*"/./"*|"//"*|*"//"*) printf '%s\n' 'error: source preflight report path is unsafe' >&2; exit 2 ;;
esac
SOURCE_PREFLIGHT_REPORT=$(canonical_nonsymlink_file "$CANDIDATE_EVIDENCE_ROOT/$SOURCE_PREFLIGHT_RELATIVE")
[[ "$SOURCE_PREFLIGHT_REPORT" == "$CANDIDATE_EVIDENCE_ROOT/"* ]] || { printf '%s\n' 'error: source preflight report escaped candidate evidence root' >&2; exit 2; }
[[ "$(/usr/bin/sha256sum -- "$SOURCE_PREFLIGHT_REPORT" | /usr/bin/awk '{print $1}')" == "$SOURCE_PREFLIGHT_SHA256" ]] || { printf '%s\n' 'error: source preflight report SHA-256 differs' >&2; exit 2; }
(
  cd /
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    /usr/bin/python3 -I -S "$SOURCE_TRANSITION_VERIFIER" \
    --verify "$SOURCE_PREFLIGHT_REPORT" --repo "$REPO" \
    --authority-policy "$SOURCE_TRANSITION_POLICY" \
    --authority-policy-sha256 "$SOURCE_TRANSITION_POLICY_SHA256" >/dev/null
) || { printf '%s\n' 'error: independently pinned source-transition preflight failed' >&2; exit 2; }

FINAL_PYTHON=$(canonical_nonsymlink_file "$REPO/.envs/gpmeep-v1.1-test/bin/python3.11")
FINAL_PYTHON_SHA256=$(/usr/bin/sha256sum -- "$FINAL_PYTHON" | /usr/bin/awk '{print $1}')
verify_final_runtime() {
  /usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    "$FINAL_PYTHON" -I -S "$REPO/scripts/verify-v110-final-runtime.py" \
    --repo "$REPO" --source-commit "$SOURCE_PREFLIGHT_COMMIT" \
    --python-sha256 "$FINAL_PYTHON_SHA256" >/dev/null
}
verify_final_runtime

BUNDLE="$ASSET_DIRECTORY/gpmeep-v1.1.0-complete-evidence.tar.gz"
ASSET_MANIFEST="$ASSET_DIRECTORY/gpmeep-v1.1.0-release-assets.json"
BUNDLE_INPUTS=(
  --evidence-tree "qualification=$EVIDENCE_ROOT"
  --evidence-tree "candidate=$CANDIDATE_EVIDENCE_ROOT"
  --authority "candidate_contract_gate=$CANDIDATE_CONTRACT_GATE"
  --authority "candidate_registry_gate=$CANDIDATE_REGISTRY_GATE"
  --authority "candidate_attestation=$CANDIDATE_ATTESTATION"
  --authority "m4_candidate_manifest=$M4_CANDIDATE_MANIFEST"
  --authority "source_transition_policy=$SOURCE_TRANSITION_POLICY"
  --authority "source_transition_verifier=$SOURCE_TRANSITION_VERIFIER"
  --authority "final_audit_authority=$FINAL_AUDIT_AUTHORITY"
  --authority "candidate_receipt=$CANDIDATE_RECEIPT"
  --authority "qualification_manifest=$QUALIFICATION_MANIFEST"
  --authority "final_receipt=$FINAL_RECEIPT"
)

if [[ -e "$BUNDLE" || -L "$BUNDLE" ]]; then
  BUNDLE=$(canonical_nonsymlink_file "$BUNDLE")
  /usr/bin/python3 -I -S "$ASSET_TOOL" --verify-bundle --archive "$BUNDLE" "${BUNDLE_INPUTS[@]}" >/dev/null
else
  /usr/bin/python3 -I -S "$ASSET_TOOL" --create-bundle --output "$BUNDLE" "${BUNDLE_INPUTS[@]}" >/dev/null
  BUNDLE=$(canonical_nonsymlink_file "$BUNDLE")
fi

RELEASE_ASSETS=(
  "package=$PACKAGE"
  "candidate_receipt=$CANDIDATE_RECEIPT"
  "final_receipt=$FINAL_RECEIPT"
  "qualification_manifest=$QUALIFICATION_MANIFEST"
  "complete_evidence=$BUNDLE"
)
if [[ -e "$ASSET_MANIFEST" || -L "$ASSET_MANIFEST" ]]; then
  ASSET_MANIFEST=$(canonical_nonsymlink_file "$ASSET_MANIFEST")
  ASSET_ARGUMENTS=()
  for asset in "${RELEASE_ASSETS[@]}"; do ASSET_ARGUMENTS+=(--asset "$asset"); done
  /usr/bin/python3 -I -S "$ASSET_TOOL" --verify-asset-sources --manifest "$ASSET_MANIFEST" "${ASSET_ARGUMENTS[@]}" >/dev/null
else
  ASSET_ARGUMENTS=()
  for asset in "${RELEASE_ASSETS[@]}"; do ASSET_ARGUMENTS+=(--asset "$asset"); done
  /usr/bin/python3 -I -S "$ASSET_TOOL" --create-asset-manifest --output "$ASSET_MANIFEST" "${ASSET_ARGUMENTS[@]}" >/dev/null
  ASSET_MANIFEST=$(canonical_nonsymlink_file "$ASSET_MANIFEST")
fi

PACKAGE_SHA256=$(/usr/bin/sha256sum -- "$PACKAGE" | /usr/bin/awk '{print $1}')
verify_exact_release() {
  verify_final_runtime
  "$FINAL_PYTHON" -I "$FINAL_TOOL" --verify \
    --receipt "$FINAL_RECEIPT" --package "$PACKAGE" \
    --candidate-receipt "$CANDIDATE_RECEIPT" --candidate-tool "$CANDIDATE_TOOL" \
    --contract "$CONTRACT" --inversion-schema "$INVERSION_SCHEMA" \
    --inventory "$INVENTORY" --qualification-manifest "$QUALIFICATION_MANIFEST" \
    --evidence-root "$EVIDENCE_ROOT" \
    --candidate-contract-gate "$CANDIDATE_CONTRACT_GATE" \
    --candidate-registry-gate "$CANDIDATE_REGISTRY_GATE" \
    --candidate-attestation "$CANDIDATE_ATTESTATION" \
    --m4-candidate-manifest "$M4_CANDIDATE_MANIFEST" \
    --candidate-evidence-root "$CANDIDATE_EVIDENCE_ROOT" \
    --source-transition-policy "$SOURCE_TRANSITION_POLICY" \
    --expected-source-transition-policy-sha256 "$SOURCE_TRANSITION_POLICY_SHA256" \
    --source-transition-verifier "$SOURCE_TRANSITION_VERIFIER" \
    --expected-source-transition-verifier-sha256 "$SOURCE_TRANSITION_VERIFIER_SHA256" \
    --final-audit-authority "$FINAL_AUDIT_AUTHORITY" \
    --expected-final-audit-authority-sha256 "$FINAL_AUDIT_AUTHORITY_SHA256" >/dev/null
  [[ "$(/usr/bin/sha256sum -- "$PACKAGE" | /usr/bin/awk '{print $1}')" == "$PACKAGE_SHA256" ]] || {
    printf '%s\n' 'error: package bytes changed after release verification' >&2
    exit 1
  }
}

verify_all_local() {
  local asset arguments=()
  verify_exact_release
  /usr/bin/python3 -I -S "$ASSET_TOOL" --verify-bundle --archive "$BUNDLE" "${BUNDLE_INPUTS[@]}" >/dev/null
  for asset in "${RELEASE_ASSETS[@]}"; do arguments+=(--asset "$asset"); done
  /usr/bin/python3 -I -S "$ASSET_TOOL" --verify-asset-sources --manifest "$ASSET_MANIFEST" "${arguments[@]}" >/dev/null
}
verify_all_local

RECEIPT_PACKAGE_SHA256=$(/usr/bin/jq -er '.package.sha256' "$FINAL_RECEIPT")
FINAL_RECEIPT_ID=$(/usr/bin/jq -er '.receipt_id' "$FINAL_RECEIPT")
CANDIDATE_RECEIPT_ID=$(/usr/bin/jq -er '.receipt_id' "$CANDIDATE_RECEIPT")
SOURCE_COMMIT=$(/usr/bin/jq -er '.tag_intent.target_commit' "$FINAL_RECEIPT")
TAG=$(/usr/bin/jq -er '.tag_intent.name' "$FINAL_RECEIPT")
MANIFEST_ID=$(/usr/bin/jq -er '.manifest_id' "$ASSET_MANIFEST")
for identity in "$FINAL_RECEIPT_ID" "$CANDIDATE_RECEIPT_ID" "$MANIFEST_ID"; do
  [[ "$identity" =~ ^[0-9a-f]{64}$ ]] || { printf '%s\n' 'error: release identity is not 64 lowercase hex' >&2; exit 1; }
done
TAG_MESSAGE="gpmeep $TAG
source-commit: $SOURCE_COMMIT
package-sha256: $PACKAGE_SHA256
candidate-receipt-id: $CANDIDATE_RECEIPT_ID
final-receipt-id: $FINAL_RECEIPT_ID
asset-manifest-id: $MANIFEST_ID"
[[ "$RECEIPT_PACKAGE_SHA256" == "$PACKAGE_SHA256" ]] || { printf '%s\n' 'error: final receipt package SHA-256 differs' >&2; exit 1; }
[[ "$TAG" == v1.1.0 && "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { printf '%s\n' 'error: final receipt tag intent differs' >&2; exit 1; }
[[ "$(/usr/bin/git -C "$REPO" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || { printf '%s\n' 'error: repository HEAD differs from authorized source' >&2; exit 1; }
[[ -z "$(/usr/bin/git -C "$REPO" status --porcelain --untracked-files=all)" ]] || { printf '%s\n' 'error: publication worktree is not clean' >&2; exit 1; }
ORIGIN_URL=$(/usr/bin/git -C "$REPO" remote get-url origin)
case "$ORIGIN_URL" in
  "git@github.com:${REPOSITORY}"|"git@github.com:${REPOSITORY}.git"|"https://github.com/${REPOSITORY}"|"https://github.com/${REPOSITORY}.git") ;;
  *) printf 'error: origin does not match --repository: %s\n' "$ORIGIN_URL" >&2; exit 1 ;;
esac

verify_local_tag() {
  local actual_message
  [[ "$(/usr/bin/git -C "$REPO" cat-file -t "refs/tags/$TAG")" == tag ]] || { printf '%s\n' 'error: local release tag is not annotated' >&2; exit 1; }
  [[ "$(/usr/bin/git -C "$REPO" rev-parse "$TAG^{}")" == "$SOURCE_COMMIT" ]] || { printf '%s\n' 'error: local tag target differs' >&2; exit 1; }
  actual_message=$(/usr/bin/git -C "$REPO" for-each-ref --format='%(contents)' "refs/tags/$TAG")
  [[ "$actual_message" == "$TAG_MESSAGE" ]] || { printf '%s\n' 'error: local tag identity message differs' >&2; exit 1; }
}

remote_tag_state() {
  /usr/bin/git -C "$REPO" ls-remote --tags origin "refs/tags/$TAG" "refs/tags/$TAG^{}" 2>/dev/null || true
}

verify_remote_tag() {
  local state peeled_count base_count local_tag_object base_match
  state=$(remote_tag_state)
  local_tag_object=$(/usr/bin/git -C "$REPO" rev-parse "refs/tags/$TAG")
  base_count=$(printf '%s\n' "$state" | /usr/bin/awk -v ref="refs/tags/$TAG" '$2 == ref {count++} END {print count+0}')
  base_match=$(printf '%s\n' "$state" | /usr/bin/awk -v ref="refs/tags/$TAG" -v object="$local_tag_object" '$2 == ref && $1 == object {count++} END {print count+0}')
  peeled_count=$(printf '%s\n' "$state" | /usr/bin/awk -v ref="refs/tags/$TAG^{}" -v commit="$SOURCE_COMMIT" '$2 == ref && $1 == commit {count++} END {print count+0}')
  [[ "$base_count" == 1 && "$base_match" == 1 && "$peeled_count" == 1 ]] || { printf '%s\n' 'error: remote release tag object/message is absent, ambiguous, or differs' >&2; exit 1; }
}

if [[ "$EXECUTE" != true ]]; then
  if /usr/bin/git -C "$REPO" show-ref --verify --quiet "refs/tags/$TAG"; then
    verify_local_tag
  fi
  printf 'v1.1.0 publication preflight PASS: package-sha256=%s manifest-id=%s tag=%s\n' "$PACKAGE_SHA256" "$MANIFEST_ID" "$TAG"
  exit 0
fi

command -v gh >/dev/null || { printf '%s\n' 'error: gh is required for --execute' >&2; exit 2; }
IMMUTABILITY_JSON=$(gh api \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  "/repos/$REPOSITORY/immutable-releases") || {
  printf '%s\n' 'error: immutable releases are not enabled or cannot be verified; no tag/release was created' >&2
  exit 1
}
[[ "$(/usr/bin/jq -er '.enabled' <<<"$IMMUTABILITY_JSON")" == true ]] || {
  printf '%s\n' 'error: immutable releases must be enabled before tag/release creation' >&2
  exit 1
}

REMOTE_STATE=$(remote_tag_state)
RELEASE_JSON=
if RELEASE_JSON=$(gh release view "$TAG" --repo "$REPOSITORY" --json tagName,name,isDraft,isImmutable,assets 2>/dev/null); then
  RELEASE_EXISTS=true
else
  RELEASE_EXISTS=false
fi

if [[ "$RESUME" != true ]]; then
  [[ -z "$REMOTE_STATE" ]] || { printf '%s\n' 'error: remote tag exists; use --resume only after verifying the interrupted run' >&2; exit 1; }
  [[ "$RELEASE_EXISTS" == false ]] || { printf '%s\n' 'error: GitHub release exists; use --resume only for the exact interrupted run' >&2; exit 1; }
  [[ ! -e "$REPO/.git/refs/tags/$TAG" ]] || { printf '%s\n' 'error: local tag exists; use --resume only for the exact interrupted run' >&2; exit 1; }
  if /usr/bin/git -C "$REPO" show-ref --verify --quiet "refs/tags/$TAG"; then
    printf '%s\n' 'error: local tag exists; use --resume only for the exact interrupted run' >&2
    exit 1
  fi
  verify_all_local
  /usr/bin/git -C "$REPO" tag -a "$TAG" "$SOURCE_COMMIT" -m "$TAG_MESSAGE"
  verify_local_tag
  verify_all_local
  /usr/bin/git -C "$REPO" push origin "refs/tags/$TAG"
  verify_remote_tag
else
  if ! /usr/bin/git -C "$REPO" show-ref --verify --quiet "refs/tags/$TAG"; then
    [[ -n "$REMOTE_STATE" ]] || { printf '%s\n' 'error: resume requested but no exact local or remote tag exists' >&2; exit 1; }
    /usr/bin/git -C "$REPO" fetch --no-tags origin "refs/tags/$TAG:refs/tags/$TAG"
  fi
  verify_local_tag
  if [[ -z "$REMOTE_STATE" ]]; then
    verify_all_local
    /usr/bin/git -C "$REPO" push origin "refs/tags/$TAG"
  fi
  verify_remote_tag
  verify_all_local
fi

if [[ "$RELEASE_EXISTS" == false ]]; then
  gh release create "$TAG" --repo "$REPOSITORY" --draft --verify-tag \
    --title "gpmeep $TAG" \
    --notes "Exact package SHA-256: $PACKAGE_SHA256\nRelease asset manifest ID: $MANIFEST_ID"
  RELEASE_JSON=$(gh release view "$TAG" --repo "$REPOSITORY" --json tagName,name,isDraft,isImmutable,assets)
fi

[[ "$(/usr/bin/jq -er '.tagName' <<<"$RELEASE_JSON")" == "$TAG" ]] || { printf '%s\n' 'error: release tag name differs' >&2; exit 1; }
[[ "$(/usr/bin/jq -er '.name' <<<"$RELEASE_JSON")" == "gpmeep $TAG" ]] || { printf '%s\n' 'error: release title differs' >&2; exit 1; }

REMOTE_IS_DRAFT=$(/usr/bin/jq -er '.isDraft' <<<"$RELEASE_JSON")
EXPECTED_ASSET_NAMES=$(/usr/bin/jq -r '.assets[].name' "$ASSET_MANIFEST")
EXPECTED_ASSET_NAMES+=$'\n'"$(/usr/bin/basename -- "$ASSET_MANIFEST")"
while IFS= read -r remote_name; do
  [[ -n "$remote_name" ]] || continue
  if ! /usr/bin/grep -Fqx -- "$remote_name" <<<"$EXPECTED_ASSET_NAMES"; then
    printf 'error: draft contains an unknown asset and will not be modified: %s\n' "$remote_name" >&2
    exit 1
  fi
done < <(/usr/bin/jq -r '.assets[].name' <<<"$RELEASE_JSON")

LOCAL_ASSET_PATHS=("$PACKAGE" "$CANDIDATE_RECEIPT" "$FINAL_RECEIPT" "$QUALIFICATION_MANIFEST" "$BUNDLE" "$ASSET_MANIFEST")
CHECK_DIRECTORY=$(/usr/bin/mktemp -d)
DOWNLOAD_DIRECTORY=$(/usr/bin/mktemp -d)
trap '/usr/bin/rm -rf -- "$CHECK_DIRECTORY" "$DOWNLOAD_DIRECTORY"' EXIT
for local_asset in "${LOCAL_ASSET_PATHS[@]}"; do
  asset_name=$(/usr/bin/basename -- "$local_asset")
  if /usr/bin/jq -e --arg name "$asset_name" '.assets[] | select(.name == $name)' <<<"$RELEASE_JSON" >/dev/null; then
    gh release download "$TAG" --repo "$REPOSITORY" --pattern "$asset_name" --dir "$CHECK_DIRECTORY"
    downloaded="$CHECK_DIRECTORY/$asset_name"
    [[ -f "$downloaded" && ! -L "$downloaded" ]] || { printf 'error: existing draft asset is unsafe: %s\n' "$asset_name" >&2; exit 1; }
    [[ "$(/usr/bin/sha256sum -- "$downloaded" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/sha256sum -- "$local_asset" | /usr/bin/awk '{print $1}')" ]] || {
      printf 'error: existing draft asset differs and will not be overwritten: %s\n' "$asset_name" >&2
      exit 1
    }
  else
    [[ "$REMOTE_IS_DRAFT" == true ]] || { printf 'error: published release is missing asset: %s\n' "$asset_name" >&2; exit 1; }
    verify_all_local
    gh release upload "$TAG" "$local_asset" --repo "$REPOSITORY"
  fi
done

verify_all_local
gh release download "$TAG" --repo "$REPOSITORY" --dir "$DOWNLOAD_DIRECTORY"
/usr/bin/python3 -I -S "$ASSET_TOOL" --verify-asset-directory \
  --manifest "$DOWNLOAD_DIRECTORY/$(/usr/bin/basename -- "$ASSET_MANIFEST")" \
  --directory "$DOWNLOAD_DIRECTORY" --expected-manifest-id "$MANIFEST_ID" >/dev/null
verify_remote_tag

if [[ "$REMOTE_IS_DRAFT" == true ]]; then
  verify_all_local
  gh release edit "$TAG" --repo "$REPOSITORY" --draft=false --verify-tag
fi

PUBLISHED_JSON=$(gh release view "$TAG" --repo "$REPOSITORY" --json tagName,name,isDraft,isImmutable,assets)
[[ "$(/usr/bin/jq -er '.isDraft' <<<"$PUBLISHED_JSON")" == false ]] || { printf '%s\n' 'error: release remained a draft' >&2; exit 1; }
[[ "$(/usr/bin/jq -er '.isImmutable' <<<"$PUBLISHED_JSON")" == true ]] || { printf '%s\n' 'error: GitHub release immutability is not enabled; enable immutable releases before publication' >&2; exit 1; }
POST_DIRECTORY=$(/usr/bin/mktemp -d)
trap '/usr/bin/rm -rf -- "$CHECK_DIRECTORY" "$DOWNLOAD_DIRECTORY" "$POST_DIRECTORY"' EXIT
gh release download "$TAG" --repo "$REPOSITORY" --dir "$POST_DIRECTORY"
/usr/bin/python3 -I -S "$ASSET_TOOL" --verify-asset-directory \
  --manifest "$POST_DIRECTORY/$(/usr/bin/basename -- "$ASSET_MANIFEST")" \
  --directory "$POST_DIRECTORY" --expected-manifest-id "$MANIFEST_ID" >/dev/null
verify_remote_tag
verify_all_local
printf 'v1.1.0 publication PASS: package-sha256=%s manifest-id=%s tag=%s\n' "$PACKAGE_SHA256" "$MANIFEST_ID" "$TAG"
