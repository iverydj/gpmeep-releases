#!/bin/bash -p
set -Eeuo pipefail
set +x
set +v
umask 077

[[ $- == *p* ]] || { printf '%s\n' 'error: protected Bash mode is required' >&2; exit 2; }

SOURCE_REPO=
SOURCE_AUTHORITY=
TARGET_PREFLIGHT=
TARGET_HISTORY_ATTESTATION=
EXTERNAL_AUDIT=
EXTERNAL_AUDIT_SHA256=
ASSET_DIRECTORY=
STATE_ROOT=
REPOSITORY=
LOCAL_BARE_REMOTE=
PRIOR_ASSETS_ROOT=
GH_BINARY=
GH_SHA256=
FINAL_VERIFIER=
FINAL_VERIFIER_SHA256=
EXECUTE=false
RESUME=false
IDENTITY_NAME=
IDENTITY_EMAIL=
IDENTITY_EPOCH=
IDENTITY_TIMEZONE=+0000
FORBIDDEN_OBJECTS=()
FINAL_VERIFIER_ARGUMENTS=()
APPROVED_GH_VERSION=2.97.0
AUDITED_GH_SHA256=84bf91e50cbcea686e7407d7b6b09f742d66498504425d4a190f816cc073bdd1
APPROVED_GH_SHA256=$AUDITED_GH_SHA256

usage() {
  printf '%s\n' "usage: $0 --source-repo PATH --source-authority FILE --target-preflight FILE --target-history-attestation FILE --external-audit FILE --external-audit-sha256 HEX --asset-directory PATH --state-root PATH --repository OWNER/REPO --identity-name NAME --identity-email EMAIL --identity-epoch UNIX_SECONDS --final-verifier FILE --final-verifier-sha256 HEX [--final-verifier-arg ARG ...] [--identity-timezone +0000] [--local-bare-remote PATH | --gh-binary FILE --gh-sha256 HEX] [--prior-assets-root PATH] [--forbid-object OID ...] [--execute [--resume]]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-repo|--source-authority|--target-preflight|--target-history-attestation|--external-audit|--external-audit-sha256|--asset-directory|--state-root|--repository|--local-bare-remote|--prior-assets-root|--gh-binary|--gh-sha256|--final-verifier|--final-verifier-sha256|--final-verifier-arg|--identity-name|--identity-email|--identity-epoch|--identity-timezone|--forbid-object)
      [[ $# -ge 2 && -n "$2" ]] || { usage; exit 2; }
      case "$1" in
        --source-repo) SOURCE_REPO=$2 ;;
        --source-authority) SOURCE_AUTHORITY=$2 ;;
        --target-preflight) TARGET_PREFLIGHT=$2 ;;
        --target-history-attestation) TARGET_HISTORY_ATTESTATION=$2 ;;
        --external-audit) EXTERNAL_AUDIT=$2 ;;
        --external-audit-sha256) EXTERNAL_AUDIT_SHA256=$2 ;;
        --asset-directory) ASSET_DIRECTORY=$2 ;;
        --state-root) STATE_ROOT=$2 ;;
        --repository) REPOSITORY=$2 ;;
        --local-bare-remote) LOCAL_BARE_REMOTE=$2 ;;
        --prior-assets-root) PRIOR_ASSETS_ROOT=$2 ;;
        --gh-binary) GH_BINARY=$2 ;;
        --gh-sha256) GH_SHA256=$2 ;;
        --final-verifier) FINAL_VERIFIER=$2 ;;
        --final-verifier-sha256) FINAL_VERIFIER_SHA256=$2 ;;
        --final-verifier-arg) FINAL_VERIFIER_ARGUMENTS+=("$2") ;;
        --identity-name) IDENTITY_NAME=$2 ;;
        --identity-email) IDENTITY_EMAIL=$2 ;;
        --identity-epoch) IDENTITY_EPOCH=$2 ;;
        --identity-timezone) IDENTITY_TIMEZONE=$2 ;;
        --forbid-object) FORBIDDEN_OBJECTS+=("$2") ;;
      esac
      shift 2
      ;;
    --execute) EXECUTE=true; shift ;;
    --resume) RESUME=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

for value in "$SOURCE_REPO" "$SOURCE_AUTHORITY" "$TARGET_PREFLIGHT" "$TARGET_HISTORY_ATTESTATION" "$EXTERNAL_AUDIT" "$EXTERNAL_AUDIT_SHA256" "$ASSET_DIRECTORY" "$STATE_ROOT" "$REPOSITORY" "$IDENTITY_NAME" "$IDENTITY_EMAIL" "$IDENTITY_EPOCH" "$FINAL_VERIFIER" "$FINAL_VERIFIER_SHA256"; do
  [[ -n "$value" ]] || { usage; exit 2; }
done
[[ "$RESUME" != true || "$EXECUTE" == true ]] || { printf '%s\n' 'error: --resume requires --execute' >&2; exit 2; }
[[ -z "$LOCAL_BARE_REMOTE" || ( -z "$GH_BINARY" && -z "$GH_SHA256" ) ]] || { printf '%s\n' 'error: choose only local-bare or pinned GitHub transport' >&2; exit 2; }
[[ -n "$LOCAL_BARE_REMOTE" || ( -n "$GH_BINARY" && -n "$GH_SHA256" ) ]] || { printf '%s\n' 'error: production transport requires --gh-binary and --gh-sha256' >&2; exit 2; }
[[ "$REPOSITORY" == example-organization/gpmeep-releases ]] || { printf '%s\n' 'error: release-only target repository differs' >&2; exit 2; }
[[ "$IDENTITY_EPOCH" =~ ^[0-9]+$ ]] || { printf '%s\n' 'error: identity epoch is invalid' >&2; exit 2; }
[[ "$IDENTITY_TIMEZONE" == +0000 ]] || { printf '%s\n' 'error: identity timezone must be +0000' >&2; exit 2; }

canonical_file() {
  local lexical resolved
  lexical=$(/usr/bin/realpath --canonicalize-missing --no-symlinks -- "$1")
  [[ -f "$lexical" && ! -L "$lexical" ]] || { printf 'error: unsafe file: %s\n' "$1" >&2; exit 2; }
  resolved=$(/usr/bin/realpath -- "$lexical")
  [[ "$resolved" == "$lexical" ]] || { printf 'error: file path contains a symlink: %s\n' "$1" >&2; exit 2; }
  printf '%s' "$resolved"
}

canonical_directory() {
  local lexical resolved
  lexical=$(/usr/bin/realpath --canonicalize-missing --no-symlinks -- "$1")
  [[ -d "$lexical" && ! -L "$lexical" ]] || { printf 'error: unsafe directory: %s\n' "$1" >&2; exit 2; }
  resolved=$(/usr/bin/realpath -- "$lexical")
  [[ "$resolved" == "$lexical" ]] || { printf 'error: directory path contains a symlink: %s\n' "$1" >&2; exit 2; }
  printf '%s' "$resolved"
}

SOURCE_REPO=$(canonical_directory "$SOURCE_REPO")
SOURCE_AUTHORITY=$(canonical_file "$SOURCE_AUTHORITY")
TARGET_PREFLIGHT=$(canonical_file "$TARGET_PREFLIGHT")
TARGET_HISTORY_ATTESTATION=$(canonical_file "$TARGET_HISTORY_ATTESTATION")
EXTERNAL_AUDIT=$(canonical_file "$EXTERNAL_AUDIT")
[[ "$EXTERNAL_AUDIT_SHA256" =~ ^[0-9a-f]{64}$ && "$(/usr/bin/sha256sum -- "$EXTERNAL_AUDIT" | /usr/bin/awk '{print $1}')" == "$EXTERNAL_AUDIT_SHA256" ]] || { printf '%s\n' 'error: independent mirror audit differs from the out-of-band trusted digest' >&2; exit 2; }
FINAL_VERIFIER=$(canonical_file "$FINAL_VERIFIER")
[[ "$FINAL_VERIFIER_SHA256" =~ ^[0-9a-f]{64}$ && "$(/usr/bin/sha256sum -- "$FINAL_VERIFIER" | /usr/bin/awk '{print $1}')" == "$FINAL_VERIFIER_SHA256" ]] || { printf '%s\n' 'error: final verifier pin differs' >&2; exit 2; }
ASSET_DIRECTORY=$(canonical_directory "$ASSET_DIRECTORY")
STATE_ROOT=$(canonical_directory "$STATE_ROOT")
if [[ -n "$PRIOR_ASSETS_ROOT" ]]; then PRIOR_ASSETS_ROOT=$(canonical_directory "$PRIOR_ASSETS_ROOT"); fi
if [[ -n "$LOCAL_BARE_REMOTE" ]]; then LOCAL_BARE_REMOTE=$(canonical_directory "$LOCAL_BARE_REMOTE"); fi
if [[ -n "$GH_BINARY" ]]; then
  GH_BINARY=$(canonical_file "$GH_BINARY")
  [[ -x "$GH_BINARY" && "$GH_SHA256" == "$APPROVED_GH_SHA256" && "$(/usr/bin/sha256sum -- "$GH_BINARY" | /usr/bin/awk '{print $1}')" == "$APPROVED_GH_SHA256" ]] || { printf '%s\n' 'error: pinned gh binary is not the audited approved GitHub CLI' >&2; exit 2; }
fi
[[ "$(/usr/bin/stat -c '%a' -- "$STATE_ROOT")" == 700 ]] || { printf '%s\n' 'error: state root mode must be 0700' >&2; exit 2; }
case "$STATE_ROOT/" in
  "$SOURCE_REPO/"*|"$ASSET_DIRECTORY/"*) printf '%s\n' 'error: state root must be outside source and asset directories' >&2; exit 2 ;;
esac
case "$ASSET_DIRECTORY/" in
  "$SOURCE_REPO/"*) printf '%s\n' 'error: asset directory must be outside source repository' >&2; exit 2 ;;
esac
[[ "$(/usr/bin/dirname -- "$EXTERNAL_AUDIT")" == "$ASSET_DIRECTORY" ]] || { printf '%s\n' 'error: external audit must be a release asset' >&2; exit 2; }
[[ "$(/usr/bin/stat -c '%a' -- "$EXTERNAL_AUDIT")" == 600 ]] || { printf '%s\n' 'error: external audit mode must be 0600' >&2; exit 2; }

SCRIPT_DIRECTORY=$(/usr/bin/dirname -- "$0")
SCRIPT_DIRECTORY=$(canonical_directory "$SCRIPT_DIRECTORY")
PUBLISHER_FILE=$(canonical_file "$0")
AUDITED_TOOL=$(canonical_file "$SCRIPT_DIRECTORY/v110-release-only-receipt.py")
TOOL=$AUDITED_TOOL
MIRROR_TEST=$(canonical_file "$SCRIPT_DIRECTORY/tests/test_v110_release_only_mirror.py")
PYTHON=(/usr/bin/python3 -I -S "$TOOL")
GIT=(/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 XDG_CONFIG_HOME=/nonexistent GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null GIT_ATTR_NOSYSTEM=1 GIT_NO_REPLACE_OBJECTS=1 GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false GIT_SSH_VARIANT=ssh "GIT_SSH_COMMAND=/usr/bin/ssh -F /dev/null -o BatchMode=yes -o ClearAllForwardings=yes" /usr/bin/git -c core.hooksPath=/dev/null)
SOURCE_REPLAY_ARGS=(--source-repo "$SOURCE_REPO" --final-verifier "$FINAL_VERIFIER" --final-verifier-sha256 "$FINAL_VERIFIER_SHA256")
for verifier_argument in "${FINAL_VERIFIER_ARGUMENTS[@]}"; do
  SOURCE_REPLAY_ARGS+=(--final-verifier-arg "$verifier_argument")
done

GH_TOKEN_VALUE=
if [[ -z "$LOCAL_BARE_REMOTE" ]]; then
  GH_TOKEN_VALUE=${GH_TOKEN:-${GITHUB_TOKEN:-}}
  [[ -n "$GH_TOKEN_VALUE" ]] || { printf '%s\n' 'error: production GitHub transport requires GH_TOKEN or GITHUB_TOKEN' >&2; exit 2; }
fi

gh_call() (
  token_value=$GH_TOKEN_VALUE
  while IFS= read -r -d '' environment_entry; do
    environment_name=${environment_entry%%=*}
    unset "$environment_name" 2>/dev/null || true
  done < <(/usr/bin/env -0)
  export PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 XDG_CONFIG_HOME=/nonexistent
  export GH_CONFIG_DIR=/nonexistent GH_PAGER=cat NO_COLOR=1 GIT_TERMINAL_PROMPT=0
  export GH_TOKEN=$token_value
  exec "$GH_BINARY" "$@"
)

GH_CAPABILITIES_SHA256=
if [[ -z "$LOCAL_BARE_REMOTE" ]]; then
  GH_CAPABILITIES=$(
    gh_call release create --help
    gh_call release upload --help
    gh_call release download --help
    gh_call release edit --help
    gh_call release view --help
    gh_call release verify --help
    gh_call release verify-asset --help
  ) || { printf '%s\n' 'error: pinned gh CLI lacks required release capabilities' >&2; exit 2; }
  for capability in 'release create' 'release upload' 'release download' 'release edit' 'release view' 'release verify' 'release verify-asset' '--repo'; do
    /usr/bin/grep -Fq -- "$capability" <<<"$GH_CAPABILITIES" || { printf 'error: pinned gh CLI capability is absent: %s\n' "$capability" >&2; exit 2; }
  done
  GH_CAPABILITIES_SHA256=$(/usr/bin/sha256sum <<<"$GH_CAPABILITIES" | /usr/bin/awk '{print $1}')
fi

verify_audited_implementation() {
  [[ "$(/usr/bin/stat -c '%a' -- "$EXTERNAL_AUDIT")" == 600 \
     && "$(/usr/bin/sha256sum -- "$AUDITED_TOOL" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/jq -er '.implementation.receipt_tool_sha256' "$EXTERNAL_AUDIT")" \
     && "$(/usr/bin/sha256sum -- "$PUBLISHER_FILE" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/jq -er '.implementation.publisher_sha256' "$EXTERNAL_AUDIT")" \
     && "$(/usr/bin/sha256sum -- "$MIRROR_TEST" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/jq -er '.implementation.tests_sha256' "$EXTERNAL_AUDIT")" ]] || { printf '%s\n' 'error: implementation differs from independent mirror audit' >&2; return 1; }
}
verify_audited_implementation
[[ "$(/usr/bin/jq -er '.github_cli.version' "$EXTERNAL_AUDIT")" == "$APPROVED_GH_VERSION" \
   && "$(/usr/bin/jq -er '.github_cli.sha256' "$EXTERNAL_AUDIT")" == "$AUDITED_GH_SHA256" ]] || { printf '%s\n' 'error: independent audit does not approve the fixed GitHub CLI' >&2; exit 1; }

SOURCE_COMMIT=$(/usr/bin/jq -er '.source_commit' "$SOURCE_AUTHORITY")
SOURCE_TREE=$(/usr/bin/jq -er '.source_tree' "$SOURCE_AUTHORITY")
SOURCE_TAG_OBJECT=$(/usr/bin/jq -er '.source_tag_object' "$EXTERNAL_AUDIT")
[[ "$(/usr/bin/jq -er '.source_tag.object_id' "$SOURCE_AUTHORITY")" == "$SOURCE_TAG_OBJECT" ]] || { printf '%s\n' 'error: private source tag differs from the out-of-band trusted audit' >&2; exit 1; }
SOURCE_MANIFEST_NAME=$(/usr/bin/jq -er '.source_asset_manifest.name' "$SOURCE_AUTHORITY")
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ && "$SOURCE_TREE" =~ ^[0-9a-f]{40}$ && "$SOURCE_TAG_OBJECT" =~ ^[0-9a-f]{40}$ ]] || { printf '%s\n' 'error: source authority identity is invalid' >&2; exit 2; }
SOURCE_MANIFEST=$(canonical_file "$ASSET_DIRECTORY/$SOURCE_MANIFEST_NAME")

record_matches() {
  local jq_path=$1 path name size digest
  path=$(canonical_file "$2")
  name=$(/usr/bin/jq -er "$jq_path.name" "$SOURCE_AUTHORITY")
  size=$(/usr/bin/jq -er "$jq_path.size_bytes" "$SOURCE_AUTHORITY")
  digest=$(/usr/bin/jq -er "$jq_path.sha256" "$SOURCE_AUTHORITY")
  [[ "$(/usr/bin/basename -- "$path")" == "$name" ]] || { printf 'error: authority asset name differs: %s\n' "$name" >&2; exit 1; }
  [[ "$(/usr/bin/stat -c '%s' -- "$path")" == "$size" ]] || { printf 'error: authority asset size differs: %s\n' "$name" >&2; exit 1; }
  [[ "$(/usr/bin/sha256sum -- "$path" | /usr/bin/awk '{print $1}')" == "$digest" ]] || { printf 'error: authority asset digest differs: %s\n' "$name" >&2; exit 1; }
}

for authority_key in final_receipt candidate_receipt package source_asset_manifest; do
  authority_name=$(/usr/bin/jq -er --arg key "$authority_key" '.[$key].name' "$SOURCE_AUTHORITY")
  record_matches ".$authority_key" "$ASSET_DIRECTORY/$authority_name"
done

verify_local_source_identity() {
  [[ "$("${GIT[@]}" -C "$SOURCE_REPO" rev-parse "$SOURCE_COMMIT^{tree}")" == "$SOURCE_TREE" ]] || { printf '%s\n' 'error: source tree differs from source authority' >&2; exit 1; }
  [[ "$("${GIT[@]}" -C "$SOURCE_REPO" cat-file -t "$SOURCE_TAG_OBJECT")" == tag ]] || { printf '%s\n' 'error: private source tag is not annotated' >&2; exit 1; }
  [[ "$("${GIT[@]}" -C "$SOURCE_REPO" rev-parse 'refs/tags/v1.1.0^{}')" == "$SOURCE_COMMIT" ]] || { printf '%s\n' 'error: private source tag target differs' >&2; exit 1; }
  [[ "$("${GIT[@]}" -C "$SOURCE_REPO" rev-parse refs/tags/v1.1.0)" == "$SOURCE_TAG_OBJECT" ]] || { printf '%s\n' 'error: private source tag object differs' >&2; exit 1; }
}
verify_local_source_identity
"${PYTHON[@]}" verify-source-authority --source-authority "$SOURCE_AUTHORITY" \
  --asset-directory "$ASSET_DIRECTORY" --trusted-source-tag-object "$SOURCE_TAG_OBJECT" "${SOURCE_REPLAY_ARGS[@]}" >/dev/null

TARGET_SLUG=$(/usr/bin/jq -er '.repository.slug' "$TARGET_PREFLIGHT")
TARGET_ID=$(/usr/bin/jq -er '.repository.id' "$TARGET_PREFLIGHT")
TARGET_NODE_ID=$(/usr/bin/jq -er '.repository.node_id' "$TARGET_PREFLIGHT")
[[ "$TARGET_SLUG" == "$REPOSITORY" ]] || { printf '%s\n' 'error: target preflight repository differs' >&2; exit 1; }
TARGET_BOOTSTRAP_MODE=$(/usr/bin/jq -er '.bootstrap.mode' "$TARGET_PREFLIGHT")
[[ "$TARGET_BOOTSTRAP_MODE" == audited-prior-releases ]] || { printf '%s\n' 'error: v1.1.0 publication requires the independently audited v1.0.3 prior release' >&2; exit 1; }
TARGET_ATTESTATION_SHA256=$(/usr/bin/jq -er '.bootstrap.attestation_sha256' "$TARGET_PREFLIGHT")
TARGET_ATTESTATION_ID=$(/usr/bin/jq -er '.bootstrap.attestation_id' "$TARGET_PREFLIGHT")
[[ "$(/usr/bin/sha256sum -- "$TARGET_HISTORY_ATTESTATION" | /usr/bin/awk '{print $1}')" == "$TARGET_ATTESTATION_SHA256" ]] || { printf '%s\n' 'error: target history attestation digest differs' >&2; exit 1; }
/usr/bin/jq -e --arg slug "$TARGET_SLUG" --argjson id "$TARGET_ID" --arg node "$TARGET_NODE_ID" --arg mode "$TARGET_BOOTSTRAP_MODE" --arg attestation_id "$TARGET_ATTESTATION_ID" --arg audit_sha "$EXTERNAL_AUDIT_SHA256" '
  .schema == "gpmeep-v110-release-only-target-history-attestation-v2" and
  .state == "HISTORY_CONFIDENTIALITY_ATTESTED" and .attestation_id == $attestation_id and
  .mode == $mode and .never_imported_or_transferred == true and
  .repository == {slug:$slug,id:$id,node_id:$node} and
  .external_audit.sha256 == $audit_sha and
  .advertised_refs == ["refs/tags/v1.0.3"] and
  (.prior_releases | length) == 1 and .prior_releases[0].tag == "v1.0.3"
' "$TARGET_HISTORY_ATTESTATION" >/dev/null || { printf '%s\n' 'error: target history attestation content differs' >&2; exit 1; }
/usr/bin/jq -e '
  .advertised_refs == ["refs/tags/v1.0.3"] and
  .bootstrap.mode == "audited-prior-releases"
' "$TARGET_PREFLIGHT" >/dev/null || { printf '%s\n' 'error: v1.1.0 target preflight does not bind only the audited v1.0.3 prior tag' >&2; exit 1; }
/usr/bin/jq -e --arg audit_sha "$EXTERNAL_AUDIT_SHA256" '.external_mirror_audit.sha256 == $audit_sha and .organization_access.all_members_can_read == true and .automation == {actions_enabled:false,dependabot_security_updates_enabled:false,dependabot_version_updates_enabled:false,no_live_default_branch:true,repository_administration_read_authorized:true,source_dependabot_limits:{"github-actions":0,pip:0}}' "$TARGET_PREFLIGHT" >/dev/null || { printf '%s\n' 'error: target external audit, organization access, or automation authority differs' >&2; exit 1; }

RUN_INTENT="$STATE_ROOT/run-intent.json"
intent_args=(--source-repo "$SOURCE_REPO" --source-authority "$SOURCE_AUTHORITY" \
  --target-preflight "$TARGET_PREFLIGHT" --target-history-attestation "$TARGET_HISTORY_ATTESTATION" \
  --external-audit "$EXTERNAL_AUDIT" --external-audit-sha256 "$EXTERNAL_AUDIT_SHA256" --asset-directory "$ASSET_DIRECTORY" --state-root "$STATE_ROOT" \
  --repository "$REPOSITORY" --identity-name "$IDENTITY_NAME" --identity-email "$IDENTITY_EMAIL" \
  --identity-epoch "$IDENTITY_EPOCH" --identity-timezone "$IDENTITY_TIMEZONE")
for oid in "${FORBIDDEN_OBJECTS[@]}"; do intent_args+=(--forbid-object "$oid"); done
if [[ -n "$LOCAL_BARE_REMOTE" ]]; then
  intent_args+=(--local-bare-remote "$LOCAL_BARE_REMOTE")
else
  intent_args+=(--gh-binary "$GH_BINARY" --gh-sha256 "$GH_SHA256" --gh-capabilities-sha256 "$GH_CAPABILITIES_SHA256")
fi
if [[ -e "$RUN_INTENT" ]]; then
  [[ "$RESUME" == true ]] || { printf '%s\n' 'error: run intent exists; use --execute --resume for the exact recorded run' >&2; exit 1; }
  RUN_INTENT=$(canonical_file "$RUN_INTENT")
  "${PYTHON[@]}" verify-run-intent --intent "$RUN_INTENT" "${intent_args[@]}" >/dev/null
else
  "${PYTHON[@]}" create-run-intent --output "$RUN_INTENT" "${intent_args[@]}" >/dev/null
fi

SNAPSHOT_STATE="$STATE_ROOT/snapshot-state"
if [[ -e "$SNAPSHOT_STATE" ]]; then
  [[ "$RESUME" == true ]] || { printf '%s\n' 'error: local mirror state exists; use --execute --resume only for the exact interrupted run' >&2; exit 1; }
else
  SNAPSHOT_ATTEMPT=$(/usr/bin/mktemp -d --tmpdir="$STATE_ROOT" .snapshot-attempt.XXXXXXXX)
  /usr/bin/chmod 700 "$SNAPSHOT_ATTEMPT"
  snapshot_args=(
    prepare-snapshot --source-repo "$SOURCE_REPO" --source-commit "$SOURCE_COMMIT"
    --source-tree "$SOURCE_TREE" --source-tag v1.1.0 --source-tag-object "$SOURCE_TAG_OBJECT"
    --state-root "$SNAPSHOT_ATTEMPT" --output "$SNAPSHOT_ATTEMPT/snapshot.json"
    --identity-name "$IDENTITY_NAME" --identity-email "$IDENTITY_EMAIL"
    --identity-epoch "$IDENTITY_EPOCH" --identity-timezone "$IDENTITY_TIMEZONE"
  )
  for oid in "${FORBIDDEN_OBJECTS[@]}"; do snapshot_args+=(--forbid-object "$oid"); done
  "${PYTHON[@]}" "${snapshot_args[@]}" >/dev/null
  [[ ! -e "$SNAPSHOT_STATE" ]] || { printf '%s\n' 'error: snapshot terminal state appeared concurrently' >&2; exit 1; }
  /usr/bin/mv -T -- "$SNAPSHOT_ATTEMPT" "$SNAPSHOT_STATE"
  /usr/bin/sync -d "$STATE_ROOT"
fi

SNAPSHOT=$(canonical_file "$SNAPSHOT_STATE/snapshot.json")
SNAPSHOT_REPO=$(canonical_directory "$SNAPSHOT_STATE/isolated-snapshot.git")
[[ "$(/usr/bin/jq -er '.identity.name' "$SNAPSHOT")" == "$IDENTITY_NAME" \
   && "$(/usr/bin/jq -er '.identity.email' "$SNAPSHOT")" == "$IDENTITY_EMAIL" \
   && "$(/usr/bin/jq -er '.identity.epoch' "$SNAPSHOT")" == "$IDENTITY_EPOCH" \
   && "$(/usr/bin/jq -er '.identity.timezone' "$SNAPSHOT")" == "$IDENTITY_TIMEZONE" ]] || { printf '%s\n' 'error: resumed deterministic snapshot identity differs from command' >&2; exit 1; }
MIRROR_RECEIPT="$ASSET_DIRECTORY/gpmeep-v1.1.0-release-only-receipt.json"
if [[ -e "$MIRROR_RECEIPT" ]]; then
  [[ "$RESUME" == true ]] || { printf '%s\n' 'error: mirror receipt exists; refusing to overwrite' >&2; exit 1; }
  MIRROR_RECEIPT=$(canonical_file "$MIRROR_RECEIPT")
  "${PYTHON[@]}" verify "${SOURCE_REPLAY_ARGS[@]}" --receipt "$MIRROR_RECEIPT" --source-authority "$SOURCE_AUTHORITY" \
    --snapshot "$SNAPSHOT" --target-preflight "$TARGET_PREFLIGHT" --snapshot-repo "$SNAPSHOT_REPO" \
    --asset-directory "$ASSET_DIRECTORY" --external-audit-sha256 "$EXTERNAL_AUDIT_SHA256" >/dev/null
else
  "${PYTHON[@]}" create "${SOURCE_REPLAY_ARGS[@]}" --source-authority "$SOURCE_AUTHORITY" --snapshot "$SNAPSHOT" \
    --target-preflight "$TARGET_PREFLIGHT" --snapshot-repo "$SNAPSHOT_REPO" \
    --asset-directory "$ASSET_DIRECTORY" --external-audit-sha256 "$EXTERNAL_AUDIT_SHA256" --output "$MIRROR_RECEIPT" >/dev/null
fi

DISTRIBUTION_MANIFEST="$ASSET_DIRECTORY/gpmeep-v1.1.0-release-only-assets.json"
if [[ -e "$DISTRIBUTION_MANIFEST" ]]; then
  [[ "$RESUME" == true ]] || { printf '%s\n' 'error: distribution manifest exists; refusing to overwrite' >&2; exit 1; }
  DISTRIBUTION_MANIFEST=$(canonical_file "$DISTRIBUTION_MANIFEST")
  "${PYTHON[@]}" verify-assets --manifest "$DISTRIBUTION_MANIFEST" --receipt "$MIRROR_RECEIPT" --asset-directory "$ASSET_DIRECTORY" >/dev/null
else
  DISTRIBUTED_TARGET_HISTORY="$ASSET_DIRECTORY/gpmeep-v1.1.0-target-history-attestation.json"
  if [[ "$TARGET_HISTORY_ATTESTATION" != "$DISTRIBUTED_TARGET_HISTORY" ]]; then
    if [[ -e "$DISTRIBUTED_TARGET_HISTORY" ]]; then
      [[ "$RESUME" == true ]] || { printf '%s\n' 'error: distributed target history exists; refusing to overwrite' >&2; exit 1; }
    else
      /usr/bin/install -m 600 -- "$TARGET_HISTORY_ATTESTATION" "$DISTRIBUTED_TARGET_HISTORY"
    fi
  fi
  DISTRIBUTED_TARGET_HISTORY=$(canonical_file "$DISTRIBUTED_TARGET_HISTORY")
  [[ "$(/usr/bin/sha256sum -- "$DISTRIBUTED_TARGET_HISTORY" | /usr/bin/awk '{print $1}')" == "$TARGET_ATTESTATION_SHA256" ]] || { printf '%s\n' 'error: distributed target history differs from bootstrap authority' >&2; exit 1; }
  "${PYTHON[@]}" create-assets --source-manifest "$SOURCE_MANIFEST" --asset-directory "$ASSET_DIRECTORY" \
    --mirror-receipt "$MIRROR_RECEIPT" --target-history-attestation "$DISTRIBUTED_TARGET_HISTORY" \
    --output "$DISTRIBUTION_MANIFEST" >/dev/null
fi

SEALED_STATE="$STATE_ROOT/sealed-state"
if [[ -e "$SEALED_STATE" ]]; then
  [[ "$RESUME" == true ]] || { printf '%s\n' 'error: tag attestation exists; refusing to overwrite' >&2; exit 1; }
else
  SEALED_ATTEMPT=$(/usr/bin/mktemp -d --tmpdir="$STATE_ROOT" .sealed-attempt.XXXXXXXX)
  /usr/bin/chmod 700 "$SEALED_ATTEMPT"
  seal_args=(seal-tag --snapshot-repo "$SNAPSHOT_REPO" --state-root "$SEALED_ATTEMPT" \
    --receipt "$MIRROR_RECEIPT" --distribution-manifest "$DISTRIBUTION_MANIFEST" \
    --output "$SEALED_ATTEMPT/tag-attestation.json")
  for oid in "${FORBIDDEN_OBJECTS[@]}"; do seal_args+=(--forbid-object "$oid"); done
  "${PYTHON[@]}" "${seal_args[@]}" >/dev/null
  [[ ! -e "$SEALED_STATE" ]] || { printf '%s\n' 'error: sealed terminal state appeared concurrently' >&2; exit 1; }
  /usr/bin/mv -T -- "$SEALED_ATTEMPT" "$SEALED_STATE"
  /usr/bin/sync -d "$STATE_ROOT"
fi

TAG_ATTESTATION=$(canonical_file "$SEALED_STATE/tag-attestation.json")
PUBLISHER_REPO=$(canonical_directory "$SEALED_STATE/isolated-publisher.git")
verify_isolated_args=(verify-isolated --repo "$PUBLISHER_REPO" --receipt "$MIRROR_RECEIPT" --distribution-manifest "$DISTRIBUTION_MANIFEST" --tag-attestation "$TAG_ATTESTATION")
for oid in "${FORBIDDEN_OBJECTS[@]}"; do verify_isolated_args+=(--forbid-object "$oid"); done
"${PYTHON[@]}" "${verify_isolated_args[@]}" >/dev/null

TAG_OBJECT=$(/usr/bin/jq -er '.tag_object' "$TAG_ATTESTATION")
SNAPSHOT_COMMIT=$(/usr/bin/jq -er '.tag_target' "$TAG_ATTESTATION")
MANIFEST_ID=$(/usr/bin/jq -er '.manifest_id' "$DISTRIBUTION_MANIFEST")
PUBLICATION_ID=$(/usr/bin/jq -er '.publication_id' "$MIRROR_RECEIPT")
TAG_REF=refs/tags/v1.1.0
RELEASE_TITLE='gpmeep v1.1.0 release-only mirror'
RELEASE_NOTES="Authoritative private source commit: $SOURCE_COMMIT
Parentless distribution snapshot: $SNAPSHOT_COMMIT
Mirror publication ID: $PUBLICATION_ID
Distribution asset manifest ID: $MANIFEST_ID"
TEMPORARY_ROOT=$(/usr/bin/mktemp -d /tmp/gpmeep-v110-release-only.XXXXXXXX)
/usr/bin/chmod 700 "$TEMPORARY_ROOT"
TEMPORARY_ROOT=$(canonical_directory "$TEMPORARY_ROOT")
[[ "$TEMPORARY_ROOT" == /tmp/gpmeep-v110-release-only.* ]] || { printf '%s\n' 'error: temporary root identity differs' >&2; exit 2; }
TEMPORARY_ROOT_ID=$(/usr/bin/stat -c '%d:%i' -- "$TEMPORARY_ROOT")
cleanup_temporary_root() {
  if [[ -d "$TEMPORARY_ROOT" && ! -L "$TEMPORARY_ROOT" \
        && "$TEMPORARY_ROOT" == /tmp/gpmeep-v110-release-only.* \
        && "$(/usr/bin/stat -c '%d:%i' -- "$TEMPORARY_ROOT")" == "$TEMPORARY_ROOT_ID" ]]; then
    /usr/bin/rm -rf -- "$TEMPORARY_ROOT"
  else
    printf '%s\n' 'warning: refusing to remove changed temporary root' >&2
  fi
}
trap cleanup_temporary_root EXIT

new_temporary_directory() {
  local name=$1 path
  [[ "$name" =~ ^[a-z-]+$ ]] || { printf '%s\n' 'error: invalid temporary directory role' >&2; exit 2; }
  path="$TEMPORARY_ROOT/$name"
  /usr/bin/mkdir -m 700 -- "$path"
  printf '%s' "$path"
}

new_unique_temporary_directory() {
  local role=$1 path
  [[ "$role" =~ ^[a-z-]+$ ]] || { printf '%s\n' 'error: invalid unique temporary directory role' >&2; exit 2; }
  path=$(/usr/bin/mktemp -d --tmpdir="$TEMPORARY_ROOT" "$role.XXXXXXXX")
  /usr/bin/chmod 700 -- "$path"
  path=$(canonical_directory "$path")
  case "$path/" in "$TEMPORARY_ROOT/"*) ;; *) printf '%s\n' 'error: unique temporary directory escaped its root' >&2; exit 2 ;; esac
  printf '%s' "$path"
}

remote_refs() {
  "${GIT[@]}" ls-remote --refs "$1" | /usr/bin/sort -k2,2
}

target_ref_names() {
  /usr/bin/awk '{print $2}' | /usr/bin/sort -u
}

github_release_state() {
  local pages count expected_prior_tags actual_tags expected_with_v110
  pages=$(gh_call api --paginate --slurp -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/releases?per_page=100") || return 4
  expected_prior_tags=$(/usr/bin/jq -cS '[.prior_releases[].tag] | sort' "$TARGET_HISTORY_ATTESTATION") || return 4
  actual_tags=$(/usr/bin/jq -cS '[.[][] | .tag_name] | sort' <<<"$pages") || return 4
  count=$(/usr/bin/jq '[.[][] | select(.tag_name == "v1.1.0")] | length' <<<"$pages") || return 4
  if [[ "$count" == 0 ]]; then
    [[ "$actual_tags" == "$expected_prior_tags" ]] || return 4
    return 3
  fi
  [[ "$count" == 1 ]] || return 4
  expected_with_v110=$(/usr/bin/jq -cS '. + ["v1.1.0"] | sort' <<<"$expected_prior_tags") || return 4
  [[ "$actual_tags" == "$expected_with_v110" ]] || return 4
  /usr/bin/jq -c '[.[][] | select(.tag_name == "v1.1.0")][0] |
    {id:.id,nodeId:.node_id,tagName:.tag_name,name:.name,body:.body,isDraft:.draft,isPrerelease:.prerelease,isImmutable:.immutable,
     assets:[.assets[] | {id:.id,nodeId:.node_id,name:.name,sizeBytes:.size,digest:.digest}]}' <<<"$pages" || return 4
}

verify_release_asset_projection() {
  local release_json=$1 actual_names expected_name expected_size expected_sha asset
  actual_names=$(/usr/bin/jq -r '.assets[].name' <<<"$release_json" | /usr/bin/sort)
  [[ "$actual_names" == "$(/usr/bin/sort <<<"$EXPECTED_NAMES")" ]] || return 1
  while IFS= read -r expected_name; do
    [[ -n "$expected_name" ]] || continue
    if [[ "$expected_name" == "$(/usr/bin/basename -- "$DISTRIBUTION_MANIFEST")" ]]; then
      expected_size=$(/usr/bin/stat -c '%s' -- "$DISTRIBUTION_MANIFEST")
      expected_sha=$(/usr/bin/sha256sum -- "$DISTRIBUTION_MANIFEST" | /usr/bin/awk '{print $1}')
    else
      expected_size=$(/usr/bin/jq -er --arg name "$expected_name" '.assets[] | select(.name == $name) | .size_bytes' "$DISTRIBUTION_MANIFEST") || return 1
      expected_sha=$(/usr/bin/jq -er --arg name "$expected_name" '.assets[] | select(.name == $name) | .sha256' "$DISTRIBUTION_MANIFEST") || return 1
    fi
    asset=$(/usr/bin/jq -c --arg name "$expected_name" '[.assets[] | select(.name == $name)]' <<<"$release_json") || return 1
    [[ "$(/usr/bin/jq 'length' <<<"$asset")" == 1 \
       && "$(/usr/bin/jq -er '.[0].id' <<<"$asset")" =~ ^[1-9][0-9]*$ \
       && -n "$(/usr/bin/jq -er '.[0].nodeId' <<<"$asset")" \
       && "$(/usr/bin/jq -er '.[0].sizeBytes' <<<"$asset")" == "$expected_size" \
       && "$(/usr/bin/jq -er '.[0].digest' <<<"$asset")" == "sha256:$expected_sha" ]] || return 1
  done <<<"$EXPECTED_NAMES"
}

verify_live_target_controls() {
  local repository_json actions_json immutable_json organization_json dependabot_probe dependabot_status
  verify_audited_implementation || return 1
  [[ "$(/usr/bin/sha256sum -- "$GH_BINARY" | /usr/bin/awk '{print $1}')" == "$GH_SHA256" ]] || { printf '%s\n' 'error: pinned gh binary changed' >&2; return 1; }
  repository_json=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY") || return 1
  [[ "$(/usr/bin/jq -er '.full_name' <<<"$repository_json")" == "$REPOSITORY" \
     && "$(/usr/bin/jq -er '.id' <<<"$repository_json")" == "$TARGET_ID" \
     && "$(/usr/bin/jq -er '.node_id' <<<"$repository_json")" == "$TARGET_NODE_ID" \
     && "$(/usr/bin/jq -er '.private' <<<"$repository_json")" == true \
     && "$(/usr/bin/jq -er '.visibility' <<<"$repository_json")" == private \
     && "$(/usr/bin/jq -er '.fork' <<<"$repository_json")" == false \
     && "$(/usr/bin/jq -er '.is_template' <<<"$repository_json")" == false \
     && "$(/usr/bin/jq -er '.archived' <<<"$repository_json")" == false \
     && "$(/usr/bin/jq -er '.disabled' <<<"$repository_json")" == false \
     && "$(/usr/bin/jq -er '.permissions.admin' <<<"$repository_json")" == true ]] || return 1
  actions_json=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/actions/permissions") || return 1
  [[ "$(/usr/bin/jq -er '.enabled' <<<"$actions_json")" == false ]] || return 1
  immutable_json=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/immutable-releases") || return 1
  [[ "$(/usr/bin/jq -er '.enabled' <<<"$immutable_json")" == true ]] || return 1
  organization_json=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' '/orgs/example-organization') || return 1
  [[ "$(/usr/bin/jq -er '.login' <<<"$organization_json")" == "$(/usr/bin/jq -er '.organization_access.organization' "$TARGET_PREFLIGHT")" \
     && "$(/usr/bin/jq -er '.id' <<<"$organization_json")" == "$(/usr/bin/jq -er '.organization_access.id' "$TARGET_PREFLIGHT")" \
     && "$(/usr/bin/jq -er '.node_id' <<<"$organization_json")" == "$(/usr/bin/jq -er '.organization_access.node_id' "$TARGET_PREFLIGHT")" \
     && "$(/usr/bin/jq -er '.default_repository_permission' <<<"$organization_json")" == "$(/usr/bin/jq -er '.organization_access.default_repository_permission' "$TARGET_PREFLIGHT")" ]] || return 1
  case "$(/usr/bin/jq -er '.default_repository_permission' <<<"$organization_json")" in read|write|admin) ;; *) return 1 ;; esac
  set +e
  dependabot_probe=$(gh_call api --include -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/automated-security-fixes" 2>&1)
  dependabot_status=$?
  set -e
  [[ "$dependabot_status" != 0 ]] && /usr/bin/grep -Eq '^HTTP/[0-9.]+ 404 ' <<<"$dependabot_probe"
}

verify_live_source_authority() {
  local repository_json release_json tag_state private_download expected_names actual_names private_name private_size private_digest
  [[ -z "$LOCAL_BARE_REMOTE" ]] || return 0
  repository_json=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$PRIVATE_REPOSITORY") || return 1
  [[ "$(/usr/bin/jq -cS '{full_name,id,node_id,private,archived,disabled}' <<<"$repository_json")" == "$(/usr/bin/jq -cS '{full_name,id,node_id,private,archived,disabled}' <<<"$PRIVATE_REPOSITORY_JSON")" ]] || return 1
  release_json=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$PRIVATE_REPOSITORY/releases/tags/v1.1.0") || return 1
  [[ "$(/usr/bin/jq -cS '{id,node_id,tag_name,draft,prerelease,immutable}' <<<"$release_json")" == "$(/usr/bin/jq -cS '{id,node_id,tag_name,draft,prerelease,immutable}' <<<"$PRIVATE_RELEASE")" ]] || return 1
  tag_state=$("${GIT[@]}" ls-remote --tags "$PRIVATE_REMOTE_URL" "$TAG_REF" "$TAG_REF^{}") || return 1
  [[ "$tag_state" == "$PRIVATE_TAG_STATE" ]] || return 1
  private_download=$(new_unique_temporary_directory private-source-recheck) || return 1
  gh_call release download v1.1.0 --repo "$PRIVATE_REPOSITORY" --dir "$private_download" || return 1
  expected_names=$(/usr/bin/jq -r '.assets[].name' "$SOURCE_MANIFEST") || return 1
  expected_names+=$'\n'"$SOURCE_MANIFEST_NAME"
  actual_names=$(/usr/bin/find "$private_download" -mindepth 1 -maxdepth 1 -printf '%f\n' | /usr/bin/sort) || return 1
  [[ "$actual_names" == "$(/usr/bin/sort <<<"$expected_names")" ]] || return 1
  while IFS=$'\t' read -r private_name private_size private_digest; do
    [[ -n "$private_name" && -f "$private_download/$private_name" && ! -L "$private_download/$private_name" ]] || return 1
    [[ "$(/usr/bin/stat -c '%s' -- "$private_download/$private_name")" == "$private_size" \
       && "$(/usr/bin/sha256sum -- "$private_download/$private_name" | /usr/bin/awk '{print $1}')" == "$private_digest" \
       && "$(/usr/bin/sha256sum -- "$ASSET_DIRECTORY/$private_name" | /usr/bin/awk '{print $1}')" == "$private_digest" ]] || return 1
  done < <(/usr/bin/jq -r '.assets[] | [.name, (.size_bytes|tostring), .sha256] | @tsv' "$SOURCE_MANIFEST")
  [[ -f "$private_download/$SOURCE_MANIFEST_NAME" && ! -L "$private_download/$SOURCE_MANIFEST_NAME" \
     && "$(/usr/bin/sha256sum -- "$private_download/$SOURCE_MANIFEST_NAME" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/sha256sum -- "$SOURCE_MANIFEST" | /usr/bin/awk '{print $1}')" ]]
}

EXPECTED_BASE_REFS=$(/usr/bin/jq -r '.advertised_refs[]' "$TARGET_PREFLIGHT")
EXPECTED_BASE_STATE=$(/usr/bin/jq -r '.prior_releases[] | [.tag_object, ("refs/tags/" + .tag)] | @tsv' "$TARGET_HISTORY_ATTESTATION" | /usr/bin/sort -k2,2)

if [[ -n "$LOCAL_BARE_REMOTE" ]]; then
  [[ "$("${GIT[@]}" --git-dir "$LOCAL_BARE_REMOTE" rev-parse --is-bare-repository)" == true ]] || { printf '%s\n' 'error: local test remote is not bare' >&2; exit 1; }
  [[ "$("${GIT[@]}" --git-dir "$LOCAL_BARE_REMOTE" rev-parse --show-object-format)" == sha1 ]] || { printf '%s\n' 'error: local test remote object format differs' >&2; exit 1; }
  ACTIVE_REMOTE=$LOCAL_BARE_REMOTE
  REMOTE_PROTOCOL_ALLOW=always
  if [[ "$TARGET_BOOTSTRAP_MODE" == audited-prior-releases ]]; then
    [[ -n "$PRIOR_ASSETS_ROOT" ]] || { printf '%s\n' 'error: audited local prior releases require --prior-assets-root' >&2; exit 2; }
    PRIOR_AUDIT_REPO=$LOCAL_BARE_REMOTE
    if "${GIT[@]}" --git-dir "$LOCAL_BARE_REMOTE" show-ref --verify --quiet refs/tags/v1.1.0; then
      PRIOR_CLONE=$(new_temporary_directory prior-clone)
      "${GIT[@]}" -c init.defaultBranch=main init --bare --template= --object-format=sha1 "$PRIOR_CLONE/target.git" >/dev/null
      while IFS= read -r prior_tag; do
        [[ -n "$prior_tag" ]] || continue
        "${GIT[@]}" -c protocol.file.allow="$REMOTE_PROTOCOL_ALLOW" --git-dir "$PRIOR_CLONE/target.git" \
          fetch --no-write-fetch-head --no-tags "$ACTIVE_REMOTE" \
          "refs/tags/$prior_tag:refs/tags/$prior_tag" >/dev/null
      done < <(/usr/bin/jq -r '.prior_releases[].tag' "$TARGET_HISTORY_ATTESTATION")
      PRIOR_AUDIT_REPO="$PRIOR_CLONE/target.git"
    fi
    prior_args=(audit-prior --repo "$PRIOR_AUDIT_REPO" --target-preflight "$TARGET_PREFLIGHT" \
      --target-history-attestation "$TARGET_HISTORY_ATTESTATION" --prior-assets-root "$PRIOR_ASSETS_ROOT" \
      --external-audit "$EXTERNAL_AUDIT" --external-audit-sha256 "$EXTERNAL_AUDIT_SHA256")
    for oid in "${FORBIDDEN_OBJECTS[@]}"; do prior_args+=(--forbid-object "$oid"); done
    "${PYTHON[@]}" "${prior_args[@]}" >/dev/null
  fi
else
  ACTIVE_REMOTE="git@github.com:${REPOSITORY}.git"
  REMOTE_PROTOCOL_ALLOW=never
  REPOSITORY_JSON=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY") || { printf '%s\n' 'error: target repository metadata is unavailable' >&2; exit 1; }
  [[ "$(/usr/bin/jq -er '.full_name' <<<"$REPOSITORY_JSON")" == "$REPOSITORY" \
     && "$(/usr/bin/jq -er '.id' <<<"$REPOSITORY_JSON")" == "$TARGET_ID" \
     && "$(/usr/bin/jq -er '.node_id' <<<"$REPOSITORY_JSON")" == "$TARGET_NODE_ID" \
     && "$(/usr/bin/jq -er '.private' <<<"$REPOSITORY_JSON")" == true \
     && "$(/usr/bin/jq -er '.fork' <<<"$REPOSITORY_JSON")" == false \
     && "$(/usr/bin/jq -er '.is_template' <<<"$REPOSITORY_JSON")" == false \
     && "$(/usr/bin/jq -er '.archived' <<<"$REPOSITORY_JSON")" == false \
     && "$(/usr/bin/jq -er '.disabled' <<<"$REPOSITORY_JSON")" == false \
     && "$(/usr/bin/jq -er '.permissions.admin' <<<"$REPOSITORY_JSON")" == true ]] || { printf '%s\n' 'error: live target identity, repository policy, or Administration(read) authority differs' >&2; exit 1; }
  ACTIONS_JSON=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/actions/permissions") || { printf '%s\n' 'error: GitHub Actions state cannot be verified' >&2; exit 1; }
  [[ "$(/usr/bin/jq -er '.enabled' <<<"$ACTIONS_JSON")" == false ]] || { printf '%s\n' 'error: GitHub Actions must be disabled before the first ref is sent' >&2; exit 1; }
  IMMUTABLE_JSON=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/immutable-releases") || { printf '%s\n' 'error: immutable-release state cannot be verified' >&2; exit 1; }
  [[ "$(/usr/bin/jq -er '.enabled' <<<"$IMMUTABLE_JSON")" == true ]] || { printf '%s\n' 'error: immutable releases must be enabled before publication' >&2; exit 1; }
  ORGANIZATION_JSON=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' '/orgs/example-organization') || { printf '%s\n' 'error: organization access metadata is unavailable' >&2; exit 1; }
  [[ "$(/usr/bin/jq -er '.login' <<<"$ORGANIZATION_JSON")" == "$(/usr/bin/jq -er '.organization_access.organization' "$TARGET_PREFLIGHT")" \
     && "$(/usr/bin/jq -er '.id' <<<"$ORGANIZATION_JSON")" == "$(/usr/bin/jq -er '.organization_access.id' "$TARGET_PREFLIGHT")" \
     && "$(/usr/bin/jq -er '.node_id' <<<"$ORGANIZATION_JSON")" == "$(/usr/bin/jq -er '.organization_access.node_id' "$TARGET_PREFLIGHT")" \
     && "$(/usr/bin/jq -er '.default_repository_permission' <<<"$ORGANIZATION_JSON")" == "$(/usr/bin/jq -er '.organization_access.default_repository_permission' "$TARGET_PREFLIGHT")" ]] || { printf '%s\n' 'error: organization-wide read authority differs' >&2; exit 1; }
  case "$(/usr/bin/jq -er '.default_repository_permission' <<<"$ORGANIZATION_JSON")" in read|write|admin) ;; *) printf '%s\n' 'error: all organization members do not have repository read access' >&2; exit 1 ;; esac
  set +e
  DEPENDABOT_PROBE=$(gh_call api --include -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/automated-security-fixes" 2>&1)
  DEPENDABOT_STATUS=$?
  set -e
  if [[ "$DEPENDABOT_STATUS" == 0 ]]; then
    printf '%s\n' 'error: Dependabot security updates are enabled' >&2
    exit 1
  fi
  /usr/bin/grep -Eq '^HTTP/[0-9.]+ 404 ' <<<"$DEPENDABOT_PROBE" || { printf '%s\n' 'error: Dependabot security-update state is unverifiable' >&2; exit 1; }
  PRIVATE_REPOSITORY=$(/usr/bin/jq -er '.source_authority.repository.slug' "$MIRROR_RECEIPT")
  PRIVATE_REPOSITORY_ID=$(/usr/bin/jq -er '.source_authority.repository.id' "$MIRROR_RECEIPT")
  PRIVATE_REPOSITORY_NODE_ID=$(/usr/bin/jq -er '.source_authority.repository.node_id' "$MIRROR_RECEIPT")
  PRIVATE_REPOSITORY_JSON=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$PRIVATE_REPOSITORY") || { printf '%s\n' 'error: private source repository identity is unavailable' >&2; exit 1; }
  [[ "$(/usr/bin/jq -er '.full_name' <<<"$PRIVATE_REPOSITORY_JSON")" == "$PRIVATE_REPOSITORY" \
     && "$(/usr/bin/jq -er '.id' <<<"$PRIVATE_REPOSITORY_JSON")" == "$PRIVATE_REPOSITORY_ID" \
     && "$(/usr/bin/jq -er '.node_id' <<<"$PRIVATE_REPOSITORY_JSON")" == "$PRIVATE_REPOSITORY_NODE_ID" \
     && "$(/usr/bin/jq -er '.private' <<<"$PRIVATE_REPOSITORY_JSON")" == true \
     && "$(/usr/bin/jq -er '.archived' <<<"$PRIVATE_REPOSITORY_JSON")" == false \
     && "$(/usr/bin/jq -er '.disabled' <<<"$PRIVATE_REPOSITORY_JSON")" == false ]] || { printf '%s\n' 'error: private source repository identity or active state differs' >&2; exit 1; }
  PRIVATE_RELEASE=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$PRIVATE_REPOSITORY/releases/tags/v1.1.0") || { printf '%s\n' 'error: private immutable source release is unavailable' >&2; exit 1; }
  [[ "$(/usr/bin/jq -er '.tag_name' <<<"$PRIVATE_RELEASE")" == v1.1.0 \
     && "$(/usr/bin/jq -er '.draft' <<<"$PRIVATE_RELEASE")" == false \
     && "$(/usr/bin/jq -er '.prerelease' <<<"$PRIVATE_RELEASE")" == false \
     && "$(/usr/bin/jq -er '.immutable' <<<"$PRIVATE_RELEASE")" == true \
     && "$(/usr/bin/jq -er '.id' <<<"$PRIVATE_RELEASE")" == "$(/usr/bin/jq -er '.source_authority.private_release.id' "$MIRROR_RECEIPT")" \
     && "$(/usr/bin/jq -er '.node_id' <<<"$PRIVATE_RELEASE")" == "$(/usr/bin/jq -er '.source_authority.private_release.node_id' "$MIRROR_RECEIPT")" ]] || { printf '%s\n' 'error: private source release is not the exact published immutable authority' >&2; exit 1; }
  PRIVATE_REMOTE_URL="git@github.com:${PRIVATE_REPOSITORY}.git"
  PRIVATE_TAG_STATE=$("${GIT[@]}" ls-remote --tags "$PRIVATE_REMOTE_URL" "$TAG_REF" "$TAG_REF^{}")
  [[ "$(/usr/bin/awk -v ref="$TAG_REF" '$2 == ref {print $1}' <<<"$PRIVATE_TAG_STATE")" == "$SOURCE_TAG_OBJECT" \
     && "$(/usr/bin/awk -v ref="$TAG_REF^{}" '$2 == ref {print $1}' <<<"$PRIVATE_TAG_STATE")" == "$SOURCE_COMMIT" ]] || { printf '%s\n' 'error: private source remote tag identity differs' >&2; exit 1; }
  verify_live_source_authority || { printf '%s\n' 'error: private source authority or immutable release assets differ' >&2; exit 1; }
  if [[ "$TARGET_BOOTSTRAP_MODE" == audited-prior-releases ]]; then
    PRIOR_LIVE_ROOT=$(new_temporary_directory prior-live-assets)
    PRIOR_CAPTURE_ROOT=$(new_temporary_directory prior-live-captures)
    while IFS= read -r prior_tag; do
      [[ -n "$prior_tag" ]] || continue
      /usr/bin/mkdir -m 700 -- "$PRIOR_LIVE_ROOT/$prior_tag"
      /usr/bin/mkdir -m 700 -- "$PRIOR_CAPTURE_ROOT/$prior_tag"
      printf '%s\n' "$REPOSITORY_JSON" >"$PRIOR_CAPTURE_ROOT/$prior_tag/repository.json"
      printf '%s\n' "$ORGANIZATION_JSON" >"$PRIOR_CAPTURE_ROOT/$prior_tag/organization.json"
      gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/releases/tags/$prior_tag" >"$PRIOR_CAPTURE_ROOT/$prior_tag/release.json"
      gh_call release download "$prior_tag" --repo "$REPOSITORY" --dir "$PRIOR_LIVE_ROOT/$prior_tag"
      "${PYTHON[@]}" capture-v103-release-metadata \
        --receipt "$PRIOR_LIVE_ROOT/$prior_tag/gpmeep-v1.0.3-release-only-migration-receipt.json" \
        --manifest "$PRIOR_LIVE_ROOT/$prior_tag/gpmeep-v1.0.3-release-only-migration-assets.json" \
        --repository-metadata "$PRIOR_CAPTURE_ROOT/$prior_tag/repository.json" \
        --organization-metadata "$PRIOR_CAPTURE_ROOT/$prior_tag/organization.json" \
        --release-metadata "$PRIOR_CAPTURE_ROOT/$prior_tag/release.json" \
        --output "$PRIOR_LIVE_ROOT/$prior_tag/release-metadata.json" >/dev/null
      gh_call release verify "$prior_tag" --repo "$REPOSITORY" --format json >"$PRIOR_LIVE_ROOT/$prior_tag/release-attestation.json"
    done < <(/usr/bin/jq -r '.prior_releases[].tag' "$TARGET_HISTORY_ATTESTATION")
    PRIOR_CLONE=$(new_temporary_directory prior-clone)
    "${GIT[@]}" -c init.defaultBranch=main init --bare --template= --object-format=sha1 "$PRIOR_CLONE/target.git" >/dev/null
    while IFS= read -r prior_tag; do
      [[ -n "$prior_tag" ]] || continue
      "${GIT[@]}" -c protocol.file.allow="$REMOTE_PROTOCOL_ALLOW" --git-dir "$PRIOR_CLONE/target.git" \
        fetch --no-write-fetch-head --no-tags "$ACTIVE_REMOTE" \
        "refs/tags/$prior_tag:refs/tags/$prior_tag" >/dev/null
    done < <(/usr/bin/jq -r '.prior_releases[].tag' "$TARGET_HISTORY_ATTESTATION")
    prior_args=(audit-prior --repo "$PRIOR_CLONE/target.git" --target-preflight "$TARGET_PREFLIGHT" \
      --target-history-attestation "$TARGET_HISTORY_ATTESTATION" --prior-assets-root "$PRIOR_LIVE_ROOT" \
      --external-audit "$EXTERNAL_AUDIT" --external-audit-sha256 "$EXTERNAL_AUDIT_SHA256")
    for oid in "${FORBIDDEN_OBJECTS[@]}"; do prior_args+=(--forbid-object "$oid"); done
    "${PYTHON[@]}" "${prior_args[@]}" >/dev/null
  fi
fi

REMOTE_STATE=$(remote_refs "$ACTIVE_REMOTE")
LIVE_REFS=$(target_ref_names <<<"$REMOTE_STATE")
EXPECTED_POST_STATE=$({ printf '%s\n' "$EXPECTED_BASE_STATE"; printf '%s\t%s\n' "$TAG_OBJECT" "$TAG_REF"; } | /usr/bin/sed '/^$/d' | /usr/bin/sort -k2,2)
REMOTE_TAG_LINE=$(/usr/bin/awk -v ref="$TAG_REF" '$2 == ref {print $1}' <<<"$REMOTE_STATE")
REMOTE_TAG_COUNT=$(/usr/bin/awk -v ref="$TAG_REF" '$2 == ref {count++} END {print count+0}' <<<"$REMOTE_STATE")
[[ "$REMOTE_TAG_COUNT" -le 1 ]] || { printf '%s\n' 'error: remote tag state is ambiguous' >&2; exit 1; }
if [[ "$REMOTE_TAG_COUNT" == 0 ]]; then
  verify_audited_implementation
  [[ "$REMOTE_STATE" == "$EXPECTED_BASE_STATE" ]] || { printf '%s\n' 'error: target full ref union differs from the attested baseline' >&2; exit 1; }
else
  [[ "$REMOTE_STATE" == "$EXPECTED_POST_STATE" ]] || { printf '%s\n' 'error: target full ref union differs from the exact resumed release state' >&2; exit 1; }
fi
if [[ -n "$LOCAL_BARE_REMOTE" && "$TARGET_BOOTSTRAP_MODE" == empty-new-repository && "$REMOTE_TAG_COUNT" == 0 ]]; then
  "${PYTHON[@]}" verify-empty-target --repo "$LOCAL_BARE_REMOTE" >/dev/null || { printf '%s\n' 'error: empty local target contains objects or repository extensions' >&2; exit 1; }
fi
if [[ "$REMOTE_TAG_COUNT" == 1 ]]; then
  [[ "$REMOTE_TAG_LINE" == "$TAG_OBJECT" ]] || { printf '%s\n' 'error: remote tag object differs; overwrite and force are forbidden' >&2; exit 1; }
  [[ "$RESUME" == true ]] || { printf '%s\n' 'error: exact remote tag already exists; use --execute --resume for exact replay' >&2; exit 1; }
fi

RELEASE_JSON=
RELEASE_EXISTS=false
if [[ -z "$LOCAL_BARE_REMOTE" ]]; then
  set +e
  RELEASE_JSON=$(github_release_state)
  RELEASE_QUERY_STATUS=$?
  set -e
  if [[ "$RELEASE_QUERY_STATUS" == 0 ]]; then
    RELEASE_EXISTS=true
  elif [[ "$RELEASE_QUERY_STATUS" != 3 ]]; then
    printf '%s\n' 'error: target release state cannot be classified; no mutation performed' >&2
    exit 1
  fi
  if [[ "$RELEASE_EXISTS" == true ]]; then
    [[ "$REMOTE_TAG_COUNT" == 1 ]] || { printf '%s\n' 'error: organization release exists without the exact mirror tag; no mutation performed' >&2; exit 1; }
    [[ "$RESUME" == true ]] || { printf '%s\n' 'error: organization release already exists; exact --resume is required' >&2; exit 1; }
    [[ "$(/usr/bin/jq -er '.tagName' <<<"$RELEASE_JSON")" == v1.1.0 \
       && "$(/usr/bin/jq -er '.name' <<<"$RELEASE_JSON")" == "$RELEASE_TITLE" \
       && "$(/usr/bin/jq -er '.body' <<<"$RELEASE_JSON")" == "$RELEASE_NOTES" ]] || { printf '%s\n' 'error: existing release tag, title, or notes differ' >&2; exit 1; }
  fi
fi

ALLOWED_REFS=$EXPECTED_BASE_REFS
if [[ "$REMOTE_TAG_COUNT" == 1 ]]; then ALLOWED_REFS+=$'\n'"$TAG_REF"; fi
while IFS= read -r live_ref; do
  [[ -n "$live_ref" ]] || continue
  /usr/bin/grep -Fqx -- "$live_ref" <<<"$ALLOWED_REFS" || { printf 'error: target has an unexpected advertised ref: %s\n' "$live_ref" >&2; exit 1; }
done <<<"$LIVE_REFS"
while IFS= read -r expected_ref; do
  [[ -n "$expected_ref" ]] || continue
  /usr/bin/grep -Fqx -- "$expected_ref" <<<"$LIVE_REFS" || { printf 'error: target is missing an audited prior-release ref: %s\n' "$expected_ref" >&2; exit 1; }
done <<<"$EXPECTED_BASE_REFS"

# Everything above this point is a mutation-free remote preflight.  The local
# deterministic state may be built during a dry run, but neither Git refs nor
# GitHub releases/assets are changed without this explicit gate.
if [[ "$EXECUTE" != true ]]; then
  printf 'v1.1.0 release-only mirror preflight PASS: publication-id=%s manifest-id=%s tag-object=%s snapshot=%s\n' "$PUBLICATION_ID" "$MANIFEST_ID" "$TAG_OBJECT" "$SNAPSHOT_COMMIT"
  exit 0
fi

# Close the read/push window as far as Git allows: any concurrent ref or
# release transition after preflight is a hard failure, never a force retry.
verify_local_source_identity
[[ "$(/usr/bin/sha256sum -- "$TARGET_HISTORY_ATTESTATION" | /usr/bin/awk '{print $1}')" == "$TARGET_ATTESTATION_SHA256" ]] || { printf '%s\n' 'error: target history attestation changed after preflight' >&2; exit 1; }
"${PYTHON[@]}" verify "${SOURCE_REPLAY_ARGS[@]}" --receipt "$MIRROR_RECEIPT" --source-authority "$SOURCE_AUTHORITY" \
  --snapshot "$SNAPSHOT" --target-preflight "$TARGET_PREFLIGHT" --snapshot-repo "$SNAPSHOT_REPO" \
  --asset-directory "$ASSET_DIRECTORY" --external-audit-sha256 "$EXTERNAL_AUDIT_SHA256" >/dev/null
"${PYTHON[@]}" verify-assets --manifest "$DISTRIBUTION_MANIFEST" --receipt "$MIRROR_RECEIPT" --asset-directory "$ASSET_DIRECTORY" >/dev/null
"${PYTHON[@]}" "${verify_isolated_args[@]}" >/dev/null
[[ "$(remote_refs "$ACTIVE_REMOTE")" == "$REMOTE_STATE" ]] || { printf '%s\n' 'error: target refs changed after preflight; no mutation performed' >&2; exit 1; }
if [[ -z "$LOCAL_BARE_REMOTE" ]]; then
  REPOSITORY_RECHECK=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY") || { printf '%s\n' 'error: target repository identity recheck failed' >&2; exit 1; }
  [[ "$(/usr/bin/jq -er '.full_name' <<<"$REPOSITORY_RECHECK")" == "$REPOSITORY" \
     && "$(/usr/bin/jq -er '.id' <<<"$REPOSITORY_RECHECK")" == "$TARGET_ID" \
     && "$(/usr/bin/jq -er '.node_id' <<<"$REPOSITORY_RECHECK")" == "$TARGET_NODE_ID" \
     && "$(/usr/bin/jq -er '.private' <<<"$REPOSITORY_RECHECK")" == true \
     && "$(/usr/bin/jq -er '.fork' <<<"$REPOSITORY_RECHECK")" == false \
     && "$(/usr/bin/jq -er '.is_template' <<<"$REPOSITORY_RECHECK")" == false \
     && "$(/usr/bin/jq -er '.archived' <<<"$REPOSITORY_RECHECK")" == false \
     && "$(/usr/bin/jq -er '.disabled' <<<"$REPOSITORY_RECHECK")" == false \
     && "$(/usr/bin/jq -er '.permissions.admin' <<<"$REPOSITORY_RECHECK")" == true ]] || { printf '%s\n' 'error: target repository policy or Administration(read) authority changed after preflight' >&2; exit 1; }
  ACTIONS_RECHECK=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/actions/permissions") || { printf '%s\n' 'error: GitHub Actions recheck failed' >&2; exit 1; }
  [[ "$(/usr/bin/jq -er '.enabled' <<<"$ACTIONS_RECHECK")" == false ]] || { printf '%s\n' 'error: GitHub Actions became enabled after preflight' >&2; exit 1; }
  IMMUTABLE_RECHECK=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/immutable-releases") || { printf '%s\n' 'error: immutable-release recheck failed' >&2; exit 1; }
  [[ "$(/usr/bin/jq -er '.enabled' <<<"$IMMUTABLE_RECHECK")" == true ]] || { printf '%s\n' 'error: immutable releases became disabled after preflight' >&2; exit 1; }
  ORGANIZATION_RECHECK=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' '/orgs/example-organization') || { printf '%s\n' 'error: organization access recheck failed' >&2; exit 1; }
  [[ "$(/usr/bin/jq -cS '{login,id,node_id,default_repository_permission}' <<<"$ORGANIZATION_RECHECK")" == "$(/usr/bin/jq -cS '{login,id,node_id,default_repository_permission}' <<<"$ORGANIZATION_JSON")" ]] || { printf '%s\n' 'error: organization-wide read policy changed after preflight' >&2; exit 1; }
  set +e
  DEPENDABOT_RECHECK=$(gh_call api --include -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/automated-security-fixes" 2>&1)
  DEPENDABOT_RECHECK_STATUS=$?
  set -e
  [[ "$DEPENDABOT_RECHECK_STATUS" != 0 ]] && /usr/bin/grep -Eq '^HTTP/[0-9.]+ 404 ' <<<"$DEPENDABOT_RECHECK" || { printf '%s\n' 'error: Dependabot security-update policy changed after preflight' >&2; exit 1; }
  PRIVATE_RELEASE_RECHECK=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$PRIVATE_REPOSITORY/releases/tags/v1.1.0") || { printf '%s\n' 'error: private source release recheck failed' >&2; exit 1; }
  [[ "$(/usr/bin/jq -cS '{id,node_id,tag_name,draft,prerelease,immutable}' <<<"$PRIVATE_RELEASE_RECHECK")" == "$(/usr/bin/jq -cS '{id,node_id,tag_name,draft,prerelease,immutable}' <<<"$PRIVATE_RELEASE")" ]] || { printf '%s\n' 'error: private source release changed after preflight' >&2; exit 1; }
  PRIVATE_TAG_RECHECK=$("${GIT[@]}" ls-remote --tags "$PRIVATE_REMOTE_URL" "$TAG_REF" "$TAG_REF^{}")
  [[ "$PRIVATE_TAG_RECHECK" == "$PRIVATE_TAG_STATE" ]] || { printf '%s\n' 'error: private source tag changed after preflight' >&2; exit 1; }
  RELEASE_RECHECK=
  RELEASE_RECHECK_EXISTS=false
  set +e
  RELEASE_RECHECK=$(github_release_state)
  RELEASE_RECHECK_STATUS=$?
  set -e
  if [[ "$RELEASE_RECHECK_STATUS" == 0 ]]; then
    RELEASE_RECHECK_EXISTS=true
  elif [[ "$RELEASE_RECHECK_STATUS" != 3 ]]; then
    printf '%s\n' 'error: target release recheck cannot be classified; no mutation performed' >&2
    exit 1
  fi
  [[ "$RELEASE_RECHECK_EXISTS" == "$RELEASE_EXISTS" ]] || { printf '%s\n' 'error: target release state changed after preflight; no mutation performed' >&2; exit 1; }
  if [[ "$RELEASE_EXISTS" == true ]]; then
    [[ "$(/usr/bin/jq -cS . <<<"$RELEASE_RECHECK")" == "$(/usr/bin/jq -cS . <<<"$RELEASE_JSON")" ]] || { printf '%s\n' 'error: target release changed after preflight; no mutation performed' >&2; exit 1; }
  fi
fi

if [[ "$REMOTE_TAG_COUNT" == 0 ]]; then
  if [[ -z "$LOCAL_BARE_REMOTE" ]]; then
    verify_live_target_controls || { printf '%s\n' 'error: live target controls changed before tag push' >&2; exit 1; }
    verify_live_source_authority || { printf '%s\n' 'error: private source authority changed before tag push' >&2; exit 1; }
  fi
  "${PYTHON[@]}" "${verify_isolated_args[@]}" >/dev/null
  "${GIT[@]}" --git-dir "$PUBLISHER_REPO" push --porcelain "$ACTIVE_REMOTE" "$TAG_REF:$TAG_REF"
fi
[[ "$(remote_refs "$ACTIVE_REMOTE")" == "$EXPECTED_POST_STATE" ]] || { printf '%s\n' 'error: exact full remote ref union was not established after tag push' >&2; exit 1; }

REMOTE_CLOSURE=$(new_temporary_directory remote-closure)
"${GIT[@]}" init --bare --template= --object-format=sha1 "$REMOTE_CLOSURE/remote-closure.git" >/dev/null
"${GIT[@]}" -c protocol.file.allow="${REMOTE_PROTOCOL_ALLOW:-always}" --git-dir "$REMOTE_CLOSURE/remote-closure.git" fetch --no-write-fetch-head --no-tags "$ACTIVE_REMOTE" "$TAG_REF:$TAG_REF" >/dev/null
"${GIT[@]}" --git-dir "$REMOTE_CLOSURE/remote-closure.git" update-ref --no-deref HEAD "$SNAPSHOT_COMMIT"
remote_verify_args=(verify-isolated --repo "$REMOTE_CLOSURE/remote-closure.git" --receipt "$MIRROR_RECEIPT" --distribution-manifest "$DISTRIBUTION_MANIFEST" --tag-attestation "$TAG_ATTESTATION")
for oid in "${FORBIDDEN_OBJECTS[@]}"; do remote_verify_args+=(--forbid-object "$oid"); done
"${PYTHON[@]}" "${remote_verify_args[@]}" >/dev/null

if [[ -n "$LOCAL_BARE_REMOTE" ]]; then
  # Local-bare mode intentionally tests only the Git confidentiality boundary;
  # it never pretends that a filesystem directory is an immutable GitHub Release.
  "${PYTHON[@]}" "${remote_verify_args[@]}" >/dev/null
  local_remote_verify_args=(verify-published --repo "$LOCAL_BARE_REMOTE" --receipt "$MIRROR_RECEIPT" \
    --distribution-manifest "$DISTRIBUTION_MANIFEST" --asset-directory "$ASSET_DIRECTORY" \
    --target-history-attestation "$TARGET_HISTORY_ATTESTATION" --external-audit-sha256 "$EXTERNAL_AUDIT_SHA256")
  if [[ -n "$PRIOR_ASSETS_ROOT" ]]; then local_remote_verify_args+=(--prior-assets-root "$PRIOR_ASSETS_ROOT"); fi
  for oid in "${FORBIDDEN_OBJECTS[@]}"; do local_remote_verify_args+=(--forbid-object "$oid"); done
  "${PYTHON[@]}" "${local_remote_verify_args[@]}" >/dev/null
  printf 'v1.1.0 local release-only tag publication PASS: publication-id=%s tag-object=%s\n' "$PUBLICATION_ID" "$TAG_OBJECT"
  exit 0
fi

if [[ "$RELEASE_EXISTS" == false ]]; then
  [[ "$(remote_refs "$ACTIVE_REMOTE")" == "$EXPECTED_POST_STATE" ]] || { printf '%s\n' 'error: full remote ref union changed before draft creation' >&2; exit 1; }
  verify_live_target_controls || { printf '%s\n' 'error: live target controls changed before draft creation' >&2; exit 1; }
  verify_live_source_authority || { printf '%s\n' 'error: private source authority changed before draft creation' >&2; exit 1; }
  gh_call release create v1.1.0 --repo "$REPOSITORY" --draft --verify-tag --title "$RELEASE_TITLE" --notes "$RELEASE_NOTES"
  RELEASE_JSON=$(github_release_state) || { printf '%s\n' 'error: newly created draft cannot be replayed' >&2; exit 1; }
fi
[[ "$(/usr/bin/jq -er '.tagName' <<<"$RELEASE_JSON")" == v1.1.0 \
   && "$(/usr/bin/jq -er '.name' <<<"$RELEASE_JSON")" == "$RELEASE_TITLE" \
   && "$(/usr/bin/jq -er '.body' <<<"$RELEASE_JSON")" == "$RELEASE_NOTES" ]] || { printf '%s\n' 'error: existing release tag, title, or notes differ' >&2; exit 1; }

REMOTE_DRAFT=$(/usr/bin/jq -r '.isDraft' <<<"$RELEASE_JSON") || { printf '%s\n' 'error: release draft state is malformed' >&2; exit 1; }
EXPECTED_NAMES=$(/usr/bin/jq -r '.assets[].name' "$DISTRIBUTION_MANIFEST")
EXPECTED_NAMES+=$'\n'"$(/usr/bin/basename -- "$DISTRIBUTION_MANIFEST")"
while IFS= read -r remote_name; do
  [[ -n "$remote_name" ]] || continue
  /usr/bin/grep -Fqx -- "$remote_name" <<<"$EXPECTED_NAMES" || { printf 'error: release contains an unknown asset: %s\n' "$remote_name" >&2; exit 1; }
done < <(/usr/bin/jq -r '.assets[].name' <<<"$RELEASE_JSON")

CHECK_DIRECTORY=$(new_temporary_directory existing-assets)
DOWNLOAD_DIRECTORY=$(new_temporary_directory draft-download)
POST_DIRECTORY=$(new_temporary_directory published-download)

verify_download_directory() {
  local directory=$1 expected name size digest path
  expected=$(/usr/bin/jq -r '.assets[].name' "$DISTRIBUTION_MANIFEST")
  expected+=$'\n'"$(/usr/bin/basename -- "$DISTRIBUTION_MANIFEST")"
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    path=$(canonical_file "$directory/$name")
    /usr/bin/chmod 600 -- "$path"
    if [[ "$name" == "$(/usr/bin/basename -- "$DISTRIBUTION_MANIFEST")" ]]; then
      [[ "$(/usr/bin/sha256sum -- "$path" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/sha256sum -- "$DISTRIBUTION_MANIFEST" | /usr/bin/awk '{print $1}')" ]] || { printf '%s\n' 'error: downloaded distribution manifest differs' >&2; exit 1; }
    else
      size=$(/usr/bin/jq -er --arg name "$name" '.assets[] | select(.name == $name) | .size_bytes' "$DISTRIBUTION_MANIFEST")
      digest=$(/usr/bin/jq -er --arg name "$name" '.assets[] | select(.name == $name) | .sha256' "$DISTRIBUTION_MANIFEST")
      [[ "$(/usr/bin/stat -c '%s' -- "$path")" == "$size" && "$(/usr/bin/sha256sum -- "$path" | /usr/bin/awk '{print $1}')" == "$digest" ]] || { printf 'error: downloaded asset differs: %s\n' "$name" >&2; exit 1; }
    fi
  done <<<"$expected"
  ACTUAL_NAMES=$(/usr/bin/find "$directory" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | /usr/bin/sort)
  [[ "$ACTUAL_NAMES" == "$(/usr/bin/sort <<<"$expected")" ]] || { printf '%s\n' 'error: downloaded release asset inventory differs' >&2; exit 1; }
}

while IFS=$'\t' read -r name size digest; do
  [[ -n "$name" ]] || continue
  local_path=$(canonical_file "$ASSET_DIRECTORY/$name")
  [[ "$(/usr/bin/stat -c '%s' -- "$local_path")" == "$size" && "$(/usr/bin/sha256sum -- "$local_path" | /usr/bin/awk '{print $1}')" == "$digest" ]] || { printf 'error: local distribution asset differs: %s\n' "$name" >&2; exit 1; }
  if /usr/bin/jq -e --arg name "$name" '.assets[] | select(.name == $name)' <<<"$RELEASE_JSON" >/dev/null; then
    gh_call release download v1.1.0 --repo "$REPOSITORY" --pattern "$name" --dir "$CHECK_DIRECTORY"
    downloaded=$(canonical_file "$CHECK_DIRECTORY/$name")
    [[ "$(/usr/bin/stat -c '%s' -- "$downloaded")" == "$size" && "$(/usr/bin/sha256sum -- "$downloaded" | /usr/bin/awk '{print $1}')" == "$digest" ]] || { printf 'error: existing release asset differs and will not be overwritten: %s\n' "$name" >&2; exit 1; }
  else
    [[ "$REMOTE_DRAFT" == true ]] || { printf 'error: published release is missing an asset: %s\n' "$name" >&2; exit 1; }
    "${PYTHON[@]}" verify-assets --manifest "$DISTRIBUTION_MANIFEST" --receipt "$MIRROR_RECEIPT" --asset-directory "$ASSET_DIRECTORY" >/dev/null
    gh_call release upload v1.1.0 "$local_path" --repo "$REPOSITORY"
  fi
done < <(/usr/bin/jq -r '.assets[] | [.name, (.size_bytes|tostring), .sha256] | @tsv' "$DISTRIBUTION_MANIFEST")

manifest_name=$(/usr/bin/basename -- "$DISTRIBUTION_MANIFEST")
if /usr/bin/jq -e --arg name "$manifest_name" '.assets[] | select(.name == $name)' <<<"$RELEASE_JSON" >/dev/null; then
  gh_call release download v1.1.0 --repo "$REPOSITORY" --pattern "$manifest_name" --dir "$CHECK_DIRECTORY"
  [[ "$(/usr/bin/sha256sum -- "$CHECK_DIRECTORY/$manifest_name" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/sha256sum -- "$DISTRIBUTION_MANIFEST" | /usr/bin/awk '{print $1}')" ]] || { printf '%s\n' 'error: existing distribution manifest differs and will not be overwritten' >&2; exit 1; }
else
  [[ "$REMOTE_DRAFT" == true ]] || { printf '%s\n' 'error: published release is missing the distribution manifest' >&2; exit 1; }
  gh_call release upload v1.1.0 "$DISTRIBUTION_MANIFEST" --repo "$REPOSITORY"
fi

gh_call release download v1.1.0 --repo "$REPOSITORY" --dir "$DOWNLOAD_DIRECTORY"
verify_download_directory "$DOWNLOAD_DIRECTORY"
[[ "$(remote_refs "$ACTIVE_REMOTE")" == "$EXPECTED_POST_STATE" ]] || { printf '%s\n' 'error: full remote ref union changed before publication' >&2; exit 1; }
verify_live_target_controls || { printf '%s\n' 'error: live target controls changed before publication' >&2; exit 1; }
verify_live_source_authority || { printf '%s\n' 'error: private source authority changed before publication' >&2; exit 1; }

# This is the last reversible checkpoint.  Refetch the draft immediately
# before the immutable transition and reject any concurrent metadata or asset
# change rather than relying on an earlier RELEASE_JSON snapshot.
PREPUBLISH=$(github_release_state) || { printf '%s\n' 'error: prepublication release state cannot be replayed' >&2; exit 1; }
[[ "$(/usr/bin/jq -er '.id' <<<"$PREPUBLISH")" == "$(/usr/bin/jq -er '.id' <<<"$RELEASE_JSON")" \
   && "$(/usr/bin/jq -er '.nodeId' <<<"$PREPUBLISH")" == "$(/usr/bin/jq -er '.nodeId' <<<"$RELEASE_JSON")" \
   && "$(/usr/bin/jq -er '.tagName' <<<"$PREPUBLISH")" == v1.1.0 \
   && "$(/usr/bin/jq -er '.name' <<<"$PREPUBLISH")" == "$RELEASE_TITLE" \
   && "$(/usr/bin/jq -er '.body' <<<"$PREPUBLISH")" == "$RELEASE_NOTES" \
   && "$(/usr/bin/jq -er '.isPrerelease' <<<"$PREPUBLISH")" == false ]] || { printf '%s\n' 'error: release metadata changed at the prepublication checkpoint' >&2; exit 1; }
verify_release_asset_projection "$PREPUBLISH" || { printf '%s\n' 'error: release asset authority changed at the prepublication checkpoint' >&2; exit 1; }

if [[ "$REMOTE_DRAFT" == true ]]; then
  [[ "$(/usr/bin/jq -er '.isDraft' <<<"$PREPUBLISH")" == true \
     && "$(/usr/bin/jq -er '.isImmutable' <<<"$PREPUBLISH")" == false ]] || { printf '%s\n' 'error: release is not the exact mutable draft authorized for publication' >&2; exit 1; }
  gh_call release edit v1.1.0 --repo "$REPOSITORY" --draft=false --verify-tag
else
  [[ "$(/usr/bin/jq -er '.isDraft' <<<"$PREPUBLISH")" == false \
     && "$(/usr/bin/jq -er '.isImmutable' <<<"$PREPUBLISH")" == true ]] || { printf '%s\n' 'error: resumed release is not exact and immutable' >&2; exit 1; }
fi
PUBLISHED=$(github_release_state) || { printf '%s\n' 'error: published release cannot be replayed' >&2; exit 1; }
[[ "$(/usr/bin/jq -er '.id' <<<"$PUBLISHED")" == "$(/usr/bin/jq -er '.id' <<<"$PREPUBLISH")" \
   && "$(/usr/bin/jq -er '.nodeId' <<<"$PUBLISHED")" == "$(/usr/bin/jq -er '.nodeId' <<<"$PREPUBLISH")" \
   && "$(/usr/bin/jq -er '.tagName' <<<"$PUBLISHED")" == v1.1.0 \
   && "$(/usr/bin/jq -er '.name' <<<"$PUBLISHED")" == "$RELEASE_TITLE" \
   && "$(/usr/bin/jq -er '.body' <<<"$PUBLISHED")" == "$RELEASE_NOTES" \
   && "$(/usr/bin/jq -er '.isDraft' <<<"$PUBLISHED")" == false \
   && "$(/usr/bin/jq -er '.isPrerelease' <<<"$PUBLISHED")" == false \
   && "$(/usr/bin/jq -er '.isImmutable' <<<"$PUBLISHED")" == true ]] || { printf '%s\n' 'error: published organization release state differs' >&2; exit 1; }
verify_release_asset_projection "$PUBLISHED" || { printf '%s\n' 'error: published release asset authority differs' >&2; exit 1; }
gh_call release verify v1.1.0 --repo "$REPOSITORY"
while IFS= read -r signed_asset; do
  [[ -n "$signed_asset" ]] || continue
  # DOWNLOAD_DIRECTORY was populated and byte-verified before the immutable
  # publication transition.  POST_DIRECTORY is intentionally populated only
  # afterwards for an independent post-publication download replay.
  signed_asset_path=$(canonical_file "$DOWNLOAD_DIRECTORY/$signed_asset")
  gh_call release verify-asset v1.1.0 "$signed_asset_path" --repo "$REPOSITORY"
done <<<"$EXPECTED_NAMES"
gh_call release download v1.1.0 --repo "$REPOSITORY" --dir "$POST_DIRECTORY"
verify_download_directory "$POST_DIRECTORY"
[[ "$(remote_refs "$ACTIVE_REMOTE")" == "$EXPECTED_POST_STATE" ]] || { printf '%s\n' 'error: full remote ref union changed after publication' >&2; exit 1; }
verify_live_target_controls || { printf '%s\n' 'error: live target controls changed after publication' >&2; exit 1; }
verify_live_source_authority || { printf '%s\n' 'error: private source authority changed after publication' >&2; exit 1; }
"${PYTHON[@]}" "${verify_isolated_args[@]}" >/dev/null
PUBLISHED_CLONE=$(new_temporary_directory published-clone)
"${GIT[@]}" -c protocol.file.allow="$REMOTE_PROTOCOL_ALLOW" clone --mirror --template= "$ACTIVE_REMOTE" "$PUBLISHED_CLONE/target.git" >/dev/null
"${GIT[@]}" --git-dir "$PUBLISHED_CLONE/target.git" remote remove origin
published_verify_args=(verify-published --repo "$PUBLISHED_CLONE/target.git" --receipt "$MIRROR_RECEIPT" \
  --distribution-manifest "$DISTRIBUTION_MANIFEST" --asset-directory "$POST_DIRECTORY" \
  --target-history-attestation "$POST_DIRECTORY/gpmeep-v1.1.0-target-history-attestation.json" \
  --external-audit-sha256 "$EXTERNAL_AUDIT_SHA256")
if [[ "$TARGET_BOOTSTRAP_MODE" == audited-prior-releases ]]; then published_verify_args+=(--prior-assets-root "$PRIOR_LIVE_ROOT"); fi
for oid in "${FORBIDDEN_OBJECTS[@]}"; do published_verify_args+=(--forbid-object "$oid"); done
"${PYTHON[@]}" "${published_verify_args[@]}" >/dev/null
FINAL_REPOSITORY_JSON=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY") || { printf '%s\n' 'error: final target repository authority is unavailable' >&2; exit 1; }
FINAL_RELEASE_JSON=$(github_release_state) || { printf '%s\n' 'error: final release authority is unavailable' >&2; exit 1; }
[[ "$(/usr/bin/jq -cS . <<<"$FINAL_RELEASE_JSON")" == "$(/usr/bin/jq -cS . <<<"$PUBLISHED")" ]] || { printf '%s\n' 'error: final release identity or projection changed before terminal creation' >&2; exit 1; }
FINAL_REPOSITORY_FILE="$TEMPORARY_ROOT/final-repository.json"
FINAL_RELEASE_FILE="$TEMPORARY_ROOT/final-release.json"
/usr/bin/printf '%s\n' "$FINAL_REPOSITORY_JSON" >"$FINAL_REPOSITORY_FILE"
/usr/bin/printf '%s\n' "$FINAL_RELEASE_JSON" >"$FINAL_RELEASE_FILE"
/usr/bin/chmod 600 "$FINAL_REPOSITORY_FILE" "$FINAL_RELEASE_FILE"
PUBLICATION_TERMINAL="$STATE_ROOT/publication-terminal.json"
terminal_args=(--receipt "$MIRROR_RECEIPT" --distribution-manifest "$POST_DIRECTORY/gpmeep-v1.1.0-release-only-assets.json" \
  --tag-attestation "$TAG_ATTESTATION" --run-intent "$RUN_INTENT" \
  --asset-directory "$POST_DIRECTORY" --tag-object "$TAG_OBJECT" --gh-binary "$GH_BINARY" \
  --gh-capabilities-sha256 "$GH_CAPABILITIES_SHA256" --external-audit-sha256 "$EXTERNAL_AUDIT_SHA256")
if [[ -e "$PUBLICATION_TERMINAL" ]]; then
  [[ "$RESUME" == true ]] || { printf '%s\n' 'error: publication terminal exists; exact resume is required' >&2; exit 1; }
  PUBLICATION_TERMINAL=$(canonical_file "$PUBLICATION_TERMINAL")
  "${PYTHON[@]}" verify-publication-terminal --terminal "$PUBLICATION_TERMINAL" "${terminal_args[@]}" >/dev/null
else
  "${PYTHON[@]}" create-publication-terminal --output "$PUBLICATION_TERMINAL" "${terminal_args[@]}" >/dev/null
fi
TERMINAL_ID=$(/usr/bin/jq -er '.terminal_id' "$PUBLICATION_TERMINAL")
printf 'v1.1.0 release-only mirror publication PASS: publication-id=%s manifest-id=%s tag-object=%s terminal-id=%s\n' "$PUBLICATION_ID" "$MANIFEST_ID" "$TAG_OBJECT" "$TERMINAL_ID"
