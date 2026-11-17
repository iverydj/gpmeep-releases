#!/bin/bash -p
set -Eeuo pipefail
set +x
set +v
umask 077

AUDIT=
AUDIT_SHA256=
LEGACY_REPO=
LEGACY_ASSETS=
STATE_ROOT=
GH_BINARY=
GH_SHA256=
EXECUTE=false
RESUME=false
REPOSITORY=example-organization/gpmeep-releases
SOURCE_REPOSITORY=iverydj/gpmeep
TAG=v1.0.3
TITLE='gpmeep v1.0.3 release-only migration'
NOTES='Exact 18-asset v1.0.3 migration into the organization release-only repository.'
APPROVED_GH_SHA256=84bf91e50cbcea686e7407d7b6b09f742d66498504425d4a190f816cc073bdd1

usage() {
  printf '%s\n' "usage: $0 --audit FILE --audit-sha256 HEX --legacy-repo BARE --legacy-assets DIR --state-root DIR --gh-binary FILE --gh-sha256 HEX [--execute [--resume]]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit|--audit-sha256|--legacy-repo|--legacy-assets|--state-root|--gh-binary|--gh-sha256)
      [[ $# -ge 2 && -n "$2" ]] || { usage; exit 2; }
      case "$1" in
        --audit) AUDIT=$2 ;;
        --audit-sha256) AUDIT_SHA256=$2 ;;
        --legacy-repo) LEGACY_REPO=$2 ;;
        --legacy-assets) LEGACY_ASSETS=$2 ;;
        --state-root) STATE_ROOT=$2 ;;
        --gh-binary) GH_BINARY=$2 ;;
        --gh-sha256) GH_SHA256=$2 ;;
      esac
      shift 2 ;;
    --execute) EXECUTE=true; shift ;;
    --resume) RESUME=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

for value in "$AUDIT" "$AUDIT_SHA256" "$LEGACY_REPO" "$LEGACY_ASSETS" "$STATE_ROOT" "$GH_BINARY" "$GH_SHA256"; do
  [[ -n "$value" ]] || { usage; exit 2; }
done
[[ "$RESUME" != true || "$EXECUTE" == true ]] || { printf '%s\n' 'error: --resume requires --execute' >&2; exit 2; }

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

AUDIT=$(canonical_file "$AUDIT")
LEGACY_REPO=$(canonical_directory "$LEGACY_REPO")
LEGACY_ASSETS=$(canonical_directory "$LEGACY_ASSETS")
STATE_ROOT=$(canonical_directory "$STATE_ROOT")
GH_BINARY=$(canonical_file "$GH_BINARY")
paths_overlap() {
  local left=$1 right=$2
  [[ "$left" == "$right" || "$left" == "$right/"* || "$right" == "$left/"* ]]
}
for pair in \
  "$AUDIT"$'\t'"$LEGACY_REPO" \
  "$AUDIT"$'\t'"$LEGACY_ASSETS" \
  "$AUDIT"$'\t'"$STATE_ROOT" \
  "$LEGACY_REPO"$'\t'"$LEGACY_ASSETS" \
  "$LEGACY_REPO"$'\t'"$STATE_ROOT" \
  "$LEGACY_ASSETS"$'\t'"$STATE_ROOT"; do
  IFS=$'\t' read -r left right <<<"$pair"
  ! paths_overlap "$left" "$right" || { printf 'error: authority, legacy inputs, and state root must be pairwise disjoint: %s / %s\n' "$left" "$right" >&2; exit 2; }
done
[[ "$AUDIT_SHA256" =~ ^[0-9a-f]{64}$ && "$GH_SHA256" == "$APPROVED_GH_SHA256" ]] || { printf '%s\n' 'error: trusted digest syntax or approved gh digest differs' >&2; exit 2; }
AUDIT_IDENTITY=$(/usr/bin/stat -c '%d:%i:%s' -- "$AUDIT")
AUDIT_INITIAL_SHA=$(/usr/bin/sha256sum -- "$AUDIT" | /usr/bin/awk '{print $1}')
[[ "$(/usr/bin/stat -c '%a' -- "$AUDIT")" == 600 && "$AUDIT_INITIAL_SHA" == "$AUDIT_SHA256" && "$(/usr/bin/stat -c '%d:%i:%s' -- "$AUDIT")" == "$AUDIT_IDENTITY" ]] || { printf '%s\n' 'error: independent migration audit authority differs' >&2; exit 2; }
exec {GH_BINARY_FD}<"$GH_BINARY"
GH_BINARY_EXEC="/proc/self/fd/$GH_BINARY_FD"
GH_BINARY_IDENTITY=$(/usr/bin/stat -Lc '%d:%i:%s' -- "$GH_BINARY_EXEC")
[[ -x "$GH_BINARY" && "$GH_BINARY_IDENTITY" == "$(/usr/bin/stat -c '%d:%i:%s' -- "$GH_BINARY")" && "$(/usr/bin/sha256sum -- "$GH_BINARY_EXEC" | /usr/bin/awk '{print $1}')" == "$APPROVED_GH_SHA256" ]] || { printf '%s\n' 'error: GitHub CLI is not the approved binary' >&2; exit 2; }
[[ "$(/usr/bin/stat -c '%a' -- "$STATE_ROOT")" == 700 ]] || { printf '%s\n' 'error: state root mode must be 0700' >&2; exit 2; }

SCRIPT_DIRECTORY=$(canonical_directory "$(/usr/bin/dirname -- "$0")")
TOOL=$(canonical_file "$SCRIPT_DIRECTORY/v110-release-only-receipt.py")
PUBLISHER=$(canonical_file "$0")
TESTS=$(canonical_file "$SCRIPT_DIRECTORY/tests/test_v110_release_only_mirror.py")
TOOL_IDENTITY=$(/usr/bin/stat -c '%d:%i:%s' -- "$TOOL")
TOOL_SHA256=$(/usr/bin/sha256sum -- "$TOOL" | /usr/bin/awk '{print $1}')
PUBLISHER_IDENTITY=$(/usr/bin/stat -c '%d:%i:%s' -- "$PUBLISHER")
PUBLISHER_SHA256=$(/usr/bin/sha256sum -- "$PUBLISHER" | /usr/bin/awk '{print $1}')
TESTS_IDENTITY=$(/usr/bin/stat -c '%d:%i:%s' -- "$TESTS")
TESTS_SHA256=$(/usr/bin/sha256sum -- "$TESTS" | /usr/bin/awk '{print $1}')
PYTHON=(/usr/bin/python3 -I -S "$TOOL")
GIT=(/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 XDG_CONFIG_HOME=/nonexistent GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null GIT_ATTR_NOSYSTEM=1 GIT_NO_REPLACE_OBJECTS=1 GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false GIT_SSH_VARIANT=ssh "GIT_SSH_COMMAND=/usr/bin/ssh -F /dev/null -o BatchMode=yes -o ClearAllForwardings=yes" /usr/bin/git -c core.hooksPath=/dev/null)

GH_TOKEN_VALUE=${GH_TOKEN:-${GITHUB_TOKEN:-}}
[[ -n "$GH_TOKEN_VALUE" ]] || { printf '%s\n' 'error: GH_TOKEN or GITHUB_TOKEN is required' >&2; exit 2; }
gh_call() (
  token_value=$GH_TOKEN_VALUE
  before=$(/usr/bin/stat -c '%d:%i:%s' -- "$GH_BINARY") || exit 126
  digest=$(/usr/bin/sha256sum -- "$GH_BINARY_EXEC" | /usr/bin/awk '{print $1}') || exit 126
  after=$(/usr/bin/stat -c '%d:%i:%s' -- "$GH_BINARY") || exit 126
  [[ "$before" == "$GH_BINARY_IDENTITY" && "$after" == "$GH_BINARY_IDENTITY" && "$digest" == "$APPROVED_GH_SHA256" ]] || exit 126
  while IFS= read -r -d '' entry; do unset "${entry%%=*}" 2>/dev/null || true; done < <(/usr/bin/env -0)
  export PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 XDG_CONFIG_HOME=/nonexistent
  export GH_CONFIG_DIR=/nonexistent GH_PAGER=cat NO_COLOR=1 GIT_TERMINAL_PROMPT=0 GH_TOKEN=$token_value
  exec "$GH_BINARY_EXEC" "$@"
)

for command in create upload download edit view verify verify-asset; do
  gh_call release "$command" --help | /usr/bin/grep -Fq -- '--repo' || { printf 'error: approved gh lacks release %s capability\n' "$command" >&2; exit 2; }
done

TEMPORARY_ROOT=$(/usr/bin/mktemp -d /tmp/gpmeep-v103-migration.XXXXXXXX)
/usr/bin/chmod 700 "$TEMPORARY_ROOT"
TEMPORARY_ROOT=$(canonical_directory "$TEMPORARY_ROOT")
TEMPORARY_ROOT_ID=$(/usr/bin/stat -c '%d:%i' -- "$TEMPORARY_ROOT")
IMMUTABLE_REACHED=false
CURRENT_STAGE=preflight
RELEASE_ID=
RELEASE_NODE_ID=
DOWNLOAD_CLEANUP=
DOWNLOAD_CLEANUP_ID=
TERMINAL_CLEANUP=
TERMINAL_CLEANUP_ID=
cleanup() {
  if [[ -n "$DOWNLOAD_CLEANUP" && -d "$DOWNLOAD_CLEANUP" && ! -L "$DOWNLOAD_CLEANUP" && "$DOWNLOAD_CLEANUP" == "$STATE_ROOT"/.v103-download.* && "$(/usr/bin/stat -c '%d:%i' -- "$DOWNLOAD_CLEANUP")" == "$DOWNLOAD_CLEANUP_ID" ]]; then
    /usr/bin/rm -rf -- "$DOWNLOAD_CLEANUP"
  fi
  if [[ -n "$TERMINAL_CLEANUP" && -d "$TERMINAL_CLEANUP" && ! -L "$TERMINAL_CLEANUP" && "$TERMINAL_CLEANUP" == "$STATE_ROOT"/.v103-terminal.* && "$(/usr/bin/stat -c '%d:%i' -- "$TERMINAL_CLEANUP")" == "$TERMINAL_CLEANUP_ID" ]]; then
    /usr/bin/rm -rf -- "$TERMINAL_CLEANUP"
  fi
  if [[ -d "$TEMPORARY_ROOT" && ! -L "$TEMPORARY_ROOT" && "$TEMPORARY_ROOT" == /tmp/gpmeep-v103-migration.* && "$(/usr/bin/stat -c '%d:%i' -- "$TEMPORARY_ROOT")" == "$TEMPORARY_ROOT_ID" ]]; then
    /usr/bin/rm -rf -- "$TEMPORARY_ROOT"
  fi
}
write_incident() {
  local status=$1 incident intent_id
  [[ "$IMMUTABLE_REACHED" == true ]] || return 0
  incident="$STATE_ROOT/v103-migration-incident.json"
  [[ ! -e "$incident" && ! -L "$incident" ]] || return 0
  [[ -f "$STATE_ROOT/v103-migration-run-intent.json" && "$RELEASE_ID" =~ ^[1-9][0-9]*$ && -n "$RELEASE_NODE_ID" ]] || return 0
  intent_id=$(/usr/bin/jq -er '.intent_id' "$STATE_ROOT/v103-migration-run-intent.json") || return 0
  "${PYTHON[@]}" create-v103-migration-incident --stage "$CURRENT_STAGE" --exit-status "$status" \
    --observed-at-utc "$(/usr/bin/date -u +%Y-%m-%dT%H:%M:%SZ)" --audit-sha256 "$AUDIT_SHA256" \
    --intent-id "$intent_id" --release-id "$RELEASE_ID" --release-node-id "$RELEASE_NODE_ID" \
    --output "$incident" >/dev/null
}
on_exit() {
  local status=$?
  if [[ "$status" != 0 ]]; then write_incident "$status" || true; fi
  cleanup
}
trap on_exit EXIT
new_temp() { /usr/bin/mktemp -d --tmpdir="$TEMPORARY_ROOT" "$1.XXXXXXXX"; }

audit_value() { /usr/bin/jq -er "$1" "$AUDIT"; }
TARGET_ID=$(audit_value '.target.repository.id')
TARGET_NODE_ID=$(audit_value '.target.repository.node_id')
ORG_ID=$(audit_value '.target.organization.id')
ORG_NODE_ID=$(audit_value '.target.organization.node_id')
SOURCE_ID=$(audit_value '.source_authority.repository.id')
SOURCE_NODE_ID=$(audit_value '.source_authority.repository.node_id')
SOURCE_RELEASE_ID=$(audit_value '.source_authority.source_release.id')
SOURCE_RELEASE_NODE_ID=$(audit_value '.source_authority.source_release.node_id')
SOURCE_TAG_OBJECT=$(audit_value '.source_authority.source_tag.object_id')
SOURCE_COMMIT=$(audit_value '.source_authority.source_commit')
SOURCE_TREE=$(audit_value '.source_authority.source_tree')
TARGET_TAG_OBJECT=$(audit_value '.snapshot.tag_object')
TARGET_COMMIT=$(audit_value '.snapshot.snapshot_commit')

verify_audit() {
  local before after digest path identity expected_digest
  before=$(/usr/bin/stat -c '%d:%i:%s' -- "$AUDIT") || return 1
  digest=$(/usr/bin/sha256sum -- "$AUDIT" | /usr/bin/awk '{print $1}') || return 1
  after=$(/usr/bin/stat -c '%d:%i:%s' -- "$AUDIT") || return 1
  [[ "$before" == "$AUDIT_IDENTITY" && "$after" == "$AUDIT_IDENTITY" && "$digest" == "$AUDIT_SHA256" ]] || return 1
  while IFS=$'\t' read -r path identity expected_digest; do
    before=$(/usr/bin/stat -c '%d:%i:%s' -- "$path") || return 1
    digest=$(/usr/bin/sha256sum -- "$path" | /usr/bin/awk '{print $1}') || return 1
    after=$(/usr/bin/stat -c '%d:%i:%s' -- "$path") || return 1
    [[ "$before" == "$identity" && "$after" == "$identity" && "$digest" == "$expected_digest" ]] || return 1
  done <<EOF
$TOOL	$TOOL_IDENTITY	$TOOL_SHA256
$PUBLISHER	$PUBLISHER_IDENTITY	$PUBLISHER_SHA256
$TESTS	$TESTS_IDENTITY	$TESTS_SHA256
EOF
  "${PYTHON[@]}" verify-v103-migration-audit --audit "$AUDIT" --audit-sha256 "$AUDIT_SHA256" --legacy-repo "$LEGACY_REPO" --legacy-assets "$LEGACY_ASSETS" >/dev/null
}

verify_source_live() {
  local repository release ref tag_object commit download actual_names expected_names name size digest
  repository=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$SOURCE_REPOSITORY") || return 1
  [[ "$(/usr/bin/jq -cS '{full_name,id,node_id,private,archived,disabled}' <<<"$repository")" == "$(/usr/bin/jq -cnS --arg slug "$SOURCE_REPOSITORY" --argjson id "$SOURCE_ID" --arg node "$SOURCE_NODE_ID" '{full_name:$slug,id:$id,node_id:$node,private:true,archived:false,disabled:false}')" ]] || return 1
  release=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$SOURCE_REPOSITORY/releases/tags/$TAG") || return 1
  [[ "$(/usr/bin/jq -cS '{id,node_id,tag_name,draft,prerelease,immutable}' <<<"$release")" == "$(/usr/bin/jq -cnS --argjson id "$SOURCE_RELEASE_ID" --arg node "$SOURCE_RELEASE_NODE_ID" --arg tag "$TAG" '{id:$id,node_id:$node,tag_name:$tag,draft:false,prerelease:false,immutable:false}')" ]] || return 1
  [[ "$(/usr/bin/jq -cS '[.assets[]|{role:"v103_legacy_release_asset",name,size_bytes:.size,sha256:(.digest|sub("^sha256:";""))}]|sort_by(.name)' <<<"$release")" == "$(/usr/bin/jq -cS '.legacy_assets' "$AUDIT")" ]] || return 1
  ref=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$SOURCE_REPOSITORY/git/ref/tags/$TAG") || return 1
  [[ "$(/usr/bin/jq -er '.object.type' <<<"$ref")" == tag && "$(/usr/bin/jq -er '.object.sha' <<<"$ref")" == "$SOURCE_TAG_OBJECT" ]] || return 1
  tag_object=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$SOURCE_REPOSITORY/git/tags/$SOURCE_TAG_OBJECT") || return 1
  [[ "$(/usr/bin/jq -er '.object.type' <<<"$tag_object")" == commit && "$(/usr/bin/jq -er '.object.sha' <<<"$tag_object")" == "$SOURCE_COMMIT" ]] || return 1
  commit=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$SOURCE_REPOSITORY/git/commits/$SOURCE_COMMIT") || return 1
  [[ "$(/usr/bin/jq -er '.tree.sha' <<<"$commit")" == "$SOURCE_TREE" ]] || return 1
  download=$(new_temp source-download) || return 1
  gh_call release download "$TAG" --repo "$SOURCE_REPOSITORY" --dir "$download" || return 1
  actual_names=$(/usr/bin/find "$download" -mindepth 1 -maxdepth 1 -printf '%f\n' | /usr/bin/sort) || return 1
  expected_names=$(/usr/bin/jq -r '.legacy_assets[].name' "$AUDIT") || return 1
  [[ "$actual_names" == "$expected_names" ]] || return 1
  while IFS=$'\t' read -r name size digest; do
    [[ -f "$download/$name" && ! -L "$download/$name" && "$(/usr/bin/stat -c '%s' -- "$download/$name")" == "$size" && "$(/usr/bin/sha256sum -- "$download/$name" | /usr/bin/awk '{print $1}')" == "$digest" && "$(/usr/bin/sha256sum -- "$LEGACY_ASSETS/$name" | /usr/bin/awk '{print $1}')" == "$digest" ]] || return 1
  done < <(/usr/bin/jq -r '.legacy_assets[]|[.name,(.size_bytes|tostring),.sha256]|@tsv' "$AUDIT")
}

remote_refs() { "${GIT[@]}" ls-remote --refs "git@github.com:${REPOSITORY}.git" | /usr/bin/sort -k2,2; }
verify_target_controls() {
  local repository actions immutable organization dependabot
  repository=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY") || return 1
  [[ "$(/usr/bin/jq -cS '{full_name,id,node_id,private,visibility,fork,is_template,archived,disabled}' <<<"$repository")" == "$(/usr/bin/jq -cnS --arg slug "$REPOSITORY" --argjson id "$TARGET_ID" --arg node "$TARGET_NODE_ID" '{full_name:$slug,id:$id,node_id:$node,private:true,visibility:"private",fork:false,is_template:false,archived:false,disabled:false}')" && "$(/usr/bin/jq -er '.permissions.admin' <<<"$repository")" == true ]] || return 1
  actions=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/actions/permissions") || return 1
  immutable=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/immutable-releases") || return 1
  organization=$(gh_call api -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' '/orgs/example-organization') || return 1
  [[ "$(/usr/bin/jq -er '.enabled' <<<"$actions")" == false && "$(/usr/bin/jq -er '.enabled' <<<"$immutable")" == true ]] || return 1
  [[ "$(/usr/bin/jq -cS '{login,id,node_id}' <<<"$organization")" == "$(/usr/bin/jq -cnS --arg login example-organization --argjson id "$ORG_ID" --arg node "$ORG_NODE_ID" '{login:$login,id:$id,node_id:$node}')" ]] || return 1
  case "$(/usr/bin/jq -er '.default_repository_permission' <<<"$organization")" in read|write|admin) ;; *) return 1 ;; esac
  if dependabot=$(gh_call api --include -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/automated-security-fixes" 2>&1); then
    return 1
  fi
  /usr/bin/grep -Eq '^HTTP/[0-9.]+ 404 ' <<<"$dependabot"
}

release_state() {
  local pages count total
  pages=$(gh_call api --paginate --slurp -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2026-03-10' "/repos/$REPOSITORY/releases?per_page=100") || return 4
  total=$(/usr/bin/jq '[.[][]]|length' <<<"$pages") || return 4
  count=$(/usr/bin/jq '[.[][]|select(.tag_name=="v1.0.3")]|length' <<<"$pages") || return 4
  [[ "$total" == 0 && "$count" == 0 ]] && return 3
  [[ "$total" == 1 && "$count" == 1 ]] || return 4
  /usr/bin/jq -c '[.[][]|select(.tag_name=="v1.0.3")][0]' <<<"$pages"
}

verify_audit
CURRENT_STAGE=initial-live-authority
verify_source_live || { printf '%s\n' 'error: mutable v1.0.3 source authority differs at initial gate' >&2; exit 1; }
verify_target_controls || { printf '%s\n' 'error: fresh target controls differ at initial gate' >&2; exit 1; }

INITIAL_REFS=$(remote_refs)
INTENT="$STATE_ROOT/v103-migration-run-intent.json"
TARGET_CLONE="$TEMPORARY_ROOT/target.git"
"${GIT[@]}" clone --mirror --template= "git@github.com:${REPOSITORY}.git" "$TARGET_CLONE" >/dev/null
"${GIT[@]}" --git-dir "$TARGET_CLONE" remote remove origin
set +e
INITIAL_RELEASE=$(release_state)
INITIAL_RELEASE_STATUS=$?
set -e
if [[ ! -e "$INTENT" && ! -L "$INTENT" ]]; then
  [[ -z "$INITIAL_REFS" ]] || { printf '%s\n' 'error: first migration run requires an empty fresh target' >&2; exit 1; }
  [[ "$INITIAL_RELEASE_STATUS" == 3 ]] || { printf '%s\n' 'error: first migration run requires zero existing releases' >&2; exit 1; }
  "${PYTHON[@]}" create-v103-migration-run-intent --audit "$AUDIT" --audit-sha256 "$AUDIT_SHA256" --legacy-repo "$LEGACY_REPO" --legacy-assets "$LEGACY_ASSETS" --target-repo "$TARGET_CLONE" --gh-binary "$GH_BINARY" --output "$INTENT" >/dev/null
else
  [[ "$RESUME" == true && -f "$INTENT" && ! -L "$INTENT" ]] || { printf '%s\n' 'error: existing migration state requires exact --resume' >&2; exit 1; }
  [[ "$INITIAL_RELEASE_STATUS" == 0 || "$INITIAL_RELEASE_STATUS" == 3 ]] || { printf '%s\n' 'error: resumed migration release census is ambiguous' >&2; exit 1; }
  "${PYTHON[@]}" verify-v103-migration-run-intent --audit "$AUDIT" --audit-sha256 "$AUDIT_SHA256" --legacy-repo "$LEGACY_REPO" --legacy-assets "$LEGACY_ASSETS" --target-repo "$TARGET_CLONE" --gh-binary "$GH_BINARY" --intent "$INTENT" >/dev/null
fi

if [[ "$EXECUTE" != true ]]; then
  printf 'v1.0.3 migration preflight PASS: intent=%s target-id=%s\n' "$(/usr/bin/jq -er '.intent_id' "$INTENT")" "$TARGET_ID"
  exit 0
fi

CURRENT_STAGE=tag-publication
EXPECTED_TAG_STATE="$TARGET_TAG_OBJECT"$'\t'"refs/tags/$TAG"
if [[ -z "$INITIAL_REFS" ]]; then
  verify_audit; verify_source_live; verify_target_controls
  "${GIT[@]}" --git-dir "$LEGACY_REPO" push --porcelain "git@github.com:${REPOSITORY}.git" "refs/tags/$TAG:refs/tags/$TAG"
else
  [[ "$RESUME" == true && "$INITIAL_REFS" == "$EXPECTED_TAG_STATE" ]] || { printf '%s\n' 'error: resumed target ref state differs' >&2; exit 1; }
fi
[[ "$(remote_refs)" == "$EXPECTED_TAG_STATE" ]] || { printf '%s\n' 'error: exact v1.0.3 tag was not established without force' >&2; exit 1; }
verify_source_live || { printf '%s\n' 'error: mutable source changed after tag publication' >&2; exit 1; }
verify_target_controls || { printf '%s\n' 'error: target controls changed after tag publication' >&2; exit 1; }

set +e
CURRENT_STAGE=draft-discovery
RELEASE=$(release_state)
RELEASE_STATUS=$?
set -e
if [[ "$RELEASE_STATUS" == 3 ]]; then
  CURRENT_STAGE=draft-creation
  gh_call release create "$TAG" --repo "$REPOSITORY" --draft --verify-tag --title "$TITLE" --notes "$NOTES"
  RELEASE=$(release_state) || { printf '%s\n' 'error: migration draft was not created exactly' >&2; exit 1; }
elif [[ "$RELEASE_STATUS" != 0 ]]; then
  printf '%s\n' 'error: v1.0.3 target release state is ambiguous' >&2; exit 1
elif [[ "$RESUME" != true ]]; then
  printf '%s\n' 'error: existing v1.0.3 release requires --resume' >&2; exit 1
fi
[[ "$(/usr/bin/jq -er '.tag_name' <<<"$RELEASE")" == "$TAG" && "$(/usr/bin/jq -er '.name' <<<"$RELEASE")" == "$TITLE" && "$(/usr/bin/jq -er '.body' <<<"$RELEASE")" == "$NOTES" && "$(/usr/bin/jq -er '.prerelease' <<<"$RELEASE")" == false ]] || { printf '%s\n' 'error: v1.0.3 migration release metadata differs' >&2; exit 1; }

RELEASE_ID=$(/usr/bin/jq -er '.id' <<<"$RELEASE")
RELEASE_NODE_ID=$(/usr/bin/jq -er '.node_id' <<<"$RELEASE")
if [[ "$(/usr/bin/jq -er '.immutable' <<<"$RELEASE")" == true ]]; then IMMUTABLE_REACHED=true; fi
RECEIPT="$STATE_ROOT/gpmeep-v1.0.3-release-only-migration-receipt.json"
MANIFEST="$STATE_ROOT/gpmeep-v1.0.3-release-only-migration-assets.json"
materialize() {
  local output=$1; shift
  local generation_root generated generated_size generated_sha
  generation_root=$(/usr/bin/mktemp -d --tmpdir="$STATE_ROOT" .v103-materialize.XXXXXXXX)
  /usr/bin/chmod 700 "$generation_root"
  generated="$generation_root/$(/usr/bin/basename -- "$output")"
  "$@" --output "$generated" >/dev/null
  generated_size=$(/usr/bin/stat -c '%s' -- "$generated")
  generated_sha=$(/usr/bin/sha256sum -- "$generated" | /usr/bin/awk '{print $1}')
  if [[ -e "$output" || -L "$output" ]]; then
    [[ -f "$output" && ! -L "$output" ]] && /usr/bin/cmp -s -- "$output" "$generated" || { printf 'error: persisted migration evidence differs: %s\n' "$output" >&2; exit 1; }
    /usr/bin/rm -- "$generated"
  else
    /usr/bin/mv --no-clobber -- "$generated" "$output"
    [[ -f "$output" && ! -L "$output" && "$(/usr/bin/stat -c '%s' -- "$output")" == "$generated_size" && "$(/usr/bin/sha256sum -- "$output" | /usr/bin/awk '{print $1}')" == "$generated_sha" ]] || { printf 'error: persisted migration evidence raced: %s\n' "$output" >&2; exit 1; }
    if [[ -f "$generated" && ! -L "$generated" ]]; then /usr/bin/rm -- "$generated"; fi
    /usr/bin/sync -d -- "$output" "$STATE_ROOT"
  fi
  /usr/bin/rmdir -- "$generation_root"
}
materialize "$RECEIPT" "${PYTHON[@]}" create-v103-migration-receipt --audit "$AUDIT" --release-id "$RELEASE_ID" --release-node-id "$RELEASE_NODE_ID"
materialize "$MANIFEST" "${PYTHON[@]}" create-v103-migration-manifest --audit "$AUDIT" --receipt "$RECEIPT" --legacy-assets "$LEGACY_ASSETS"

STAGING="$STATE_ROOT/v103-migration-release-assets"
if [[ ! -e "$STAGING" && ! -L "$STAGING" ]]; then /usr/bin/mkdir -m 700 -- "$STAGING"; fi
STAGING=$(canonical_directory "$STAGING")
stage_file() {
  local source=$1 destination=$2 pending
  pending="$STAGING/.$(/usr/bin/basename -- "$destination").pending"
  if [[ -e "$destination" || -L "$destination" ]]; then
    [[ -f "$destination" && ! -L "$destination" ]] || return 1
    return 0
  fi
  if [[ -e "$pending" || -L "$pending" ]]; then
    [[ -f "$pending" && ! -L "$pending" && "$(/usr/bin/sha256sum -- "$pending" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/sha256sum -- "$source" | /usr/bin/awk '{print $1}')" ]] || return 1
  else
    /usr/bin/cp -- "$source" "$pending"
    /usr/bin/chmod 600 "$pending"
    /usr/bin/sync -d -- "$pending" "$STAGING"
  fi
  /usr/bin/mv --no-clobber -- "$pending" "$destination"
  [[ -f "$destination" && ! -L "$destination" && "$(/usr/bin/sha256sum -- "$destination" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/sha256sum -- "$source" | /usr/bin/awk '{print $1}')" ]] || return 1
  /usr/bin/sync -d -- "$destination" "$STAGING"
}
while IFS= read -r name; do
  source_path="$LEGACY_ASSETS/$name"
  destination="$STAGING/$name"
  stage_file "$source_path" "$destination" || { printf 'error: cannot resume staging: %s\n' "$name" >&2; exit 1; }
done < <(/usr/bin/jq -r '.legacy_assets[].name' "$AUDIT")
for source_path in "$RECEIPT" "$MANIFEST"; do
  destination="$STAGING/$(/usr/bin/basename -- "$source_path")"
  stage_file "$source_path" "$destination" || { printf 'error: cannot resume staging: %s\n' "$source_path" >&2; exit 1; }
done
EXPECTED_NAMES=$({ /usr/bin/jq -r '.legacy_assets[].name' "$AUDIT"; /usr/bin/basename -- "$RECEIPT"; /usr/bin/basename -- "$MANIFEST"; } | /usr/bin/sort)
[[ "$(/usr/bin/wc -l <<<"$EXPECTED_NAMES")" == 20 ]] || { printf '%s\n' 'error: canonical migration authority does not contain exactly 20 assets' >&2; exit 1; }
expected_with_sidecars=$({ printf '%s\n' "$EXPECTED_NAMES"; printf '%s\n' release-attestation.json release-metadata.json; } | /usr/bin/sort)

expected_size_digest() {
  local name=$1 record source
  record=$(/usr/bin/jq -er --arg name "$name" '.legacy_assets[]|select(.name==$name)|[.size_bytes,.sha256]|@tsv' "$AUDIT" 2>/dev/null) && { printf '%s' "$record"; return 0; }
  case "$name" in
    "$(/usr/bin/basename -- "$RECEIPT")") source=$RECEIPT ;;
    "$(/usr/bin/basename -- "$MANIFEST")") source=$MANIFEST ;;
    *) return 1 ;;
  esac
  printf '%s\t%s' "$(/usr/bin/stat -c '%s' -- "$source")" "$(/usr/bin/sha256sum -- "$source" | /usr/bin/awk '{print $1}')"
}

verify_staging() {
  local actual name expected_size expected_digest
  actual=$(/usr/bin/find "$STAGING" -mindepth 1 -maxdepth 1 -printf '%f\n' | /usr/bin/sort) || return 1
  [[ "$actual" == "$EXPECTED_NAMES" ]] || return 1
  while IFS= read -r name; do
    [[ -n "$name" && -f "$STAGING/$name" && ! -L "$STAGING/$name" ]] || return 1
    IFS=$'\t' read -r expected_size expected_digest <<<"$(expected_size_digest "$name")" || return 1
    [[ "$(/usr/bin/stat -c '%s' -- "$STAGING/$name")" == "$expected_size" && "$(/usr/bin/sha256sum -- "$STAGING/$name" | /usr/bin/awk '{print $1}')" == "$expected_digest" ]] || return 1
  done <<<"$EXPECTED_NAMES"
}
verify_staging || { printf '%s\n' 'error: migration staging differs from the audited canonical bytes' >&2; exit 1; }

verify_release_assets() {
  local current=$1 actual_names name expected_size expected_digest matches
  actual_names=$(/usr/bin/jq -r '.assets[].name' <<<"$current" | /usr/bin/sort)
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    /usr/bin/grep -Fqx -- "$name" <<<"$EXPECTED_NAMES" || return 1
    IFS=$'\t' read -r expected_size expected_digest <<<"$(expected_size_digest "$name")" || return 1
    matches=$(/usr/bin/jq -c --arg name "$name" '[.assets[]|select(.name==$name)]' <<<"$current") || return 1
    [[ "$(/usr/bin/jq 'length' <<<"$matches")" == 1 && "$(/usr/bin/jq -er '.[0].size' <<<"$matches")" == "$expected_size" && "$(/usr/bin/jq -er '.[0].digest' <<<"$matches")" == "sha256:$expected_digest" ]] || return 1
  done <<<"$actual_names"
}
verify_release_assets "$RELEASE" || { printf '%s\n' 'error: resumed v1.0.3 release has an unknown or changed asset' >&2; exit 1; }

if [[ "$(/usr/bin/jq -er '.immutable' <<<"$RELEASE")" == false ]]; then
  [[ "$(/usr/bin/jq -er '.draft' <<<"$RELEASE")" == true ]] || { printf '%s\n' 'error: mutable v1.0.3 release is not a resumable draft' >&2; exit 1; }
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    if ! /usr/bin/jq -e --arg name "$name" '.assets[]|select(.name==$name)' <<<"$RELEASE" >/dev/null; then
      CURRENT_STAGE="asset-upload:$name"
      verify_audit; verify_source_live; verify_target_controls; verify_staging
      [[ "$(remote_refs)" == "$EXPECTED_TAG_STATE" ]] || exit 1
      gh_call release upload "$TAG" "$STAGING/$name" --repo "$REPOSITORY"
      RELEASE=$(release_state) || exit 1
      verify_release_assets "$RELEASE" || exit 1
    fi
  done <<<"$EXPECTED_NAMES"
  [[ "$(/usr/bin/jq -r '.assets[].name' <<<"$RELEASE" | /usr/bin/sort)" == "$EXPECTED_NAMES" ]] || { printf '%s\n' 'error: migration release asset union is incomplete' >&2; exit 1; }
  CURRENT_STAGE=immutable-publication
  verify_audit; verify_source_live; verify_target_controls; verify_staging
  PREPUBLISH=$(release_state) || { printf '%s\n' 'error: v1.0.3 draft disappeared before immutable publication' >&2; exit 1; }
  [[ "$(/usr/bin/jq -cS '{id,node_id,tag_name,name,body,draft,prerelease,immutable}' <<<"$PREPUBLISH")" == "$(/usr/bin/jq -cnS --argjson id "$RELEASE_ID" --arg node "$RELEASE_NODE_ID" --arg tag "$TAG" --arg title "$TITLE" --arg body "$NOTES" '{id:$id,node_id:$node,tag_name:$tag,name:$title,body:$body,draft:true,prerelease:false,immutable:false}')" ]] || { printf '%s\n' 'error: v1.0.3 draft identity or policy raced before publish' >&2; exit 1; }
  verify_release_assets "$PREPUBLISH" || { printf '%s\n' 'error: v1.0.3 draft asset census raced before publish' >&2; exit 1; }
  [[ "$(/usr/bin/jq -r '.assets[].name' <<<"$PREPUBLISH" | /usr/bin/sort)" == "$EXPECTED_NAMES" ]] || { printf '%s\n' 'error: v1.0.3 draft asset union raced before publish' >&2; exit 1; }
  RELEASE=$PREPUBLISH
  set +e
  gh_call release edit "$TAG" --repo "$REPOSITORY" --draft=false
  PUBLISH_STATUS=$?
  set -e
  IMMUTABLE_REACHED=true
  [[ "$PUBLISH_STATUS" == 0 ]] || { printf '%s\n' 'error: immutable publication command failed or became ambiguous' >&2; exit 1; }
  RELEASE=$(release_state) || exit 1
fi

CURRENT_STAGE=postpublish-live-verification
[[ "$(/usr/bin/jq -er '.draft' <<<"$RELEASE")" == false && "$(/usr/bin/jq -er '.prerelease' <<<"$RELEASE")" == false && "$(/usr/bin/jq -er '.immutable' <<<"$RELEASE")" == true ]] || { printf '%s\n' 'error: v1.0.3 release did not become final and immutable' >&2; exit 1; }
verify_release_assets "$RELEASE" || exit 1
[[ "$(/usr/bin/jq -r '.assets[].name' <<<"$RELEASE" | /usr/bin/sort)" == "$EXPECTED_NAMES" ]] || { printf '%s\n' 'error: immutable v1.0.3 release asset union differs' >&2; exit 1; }
verify_staging || { printf '%s\n' 'error: migration staging changed after immutable publication' >&2; exit 1; }
verify_source_live || { printf '%s\n' 'error: mutable source changed after immutable publication' >&2; exit 1; }
verify_target_controls || exit 1
[[ "$(remote_refs)" == "$EXPECTED_TAG_STATE" ]] || exit 1
RELEASE=$(release_state) || { printf '%s\n' 'error: immutable v1.0.3 release census changed before download' >&2; exit 1; }
[[ "$(/usr/bin/jq -cS '{id,node_id,tag_name,name,body,draft,prerelease,immutable}' <<<"$RELEASE")" == "$(/usr/bin/jq -cnS --argjson id "$RELEASE_ID" --arg node "$RELEASE_NODE_ID" --arg tag "$TAG" --arg title "$TITLE" --arg body "$NOTES" '{id:$id,node_id:$node,tag_name:$tag,name:$title,body:$body,draft:false,prerelease:false,immutable:true}')" ]] || exit 1
verify_release_assets "$RELEASE" || exit 1
[[ "$(/usr/bin/jq -r '.assets[].name' <<<"$RELEASE" | /usr/bin/sort)" == "$EXPECTED_NAMES" ]] || exit 1

EVIDENCE="$STATE_ROOT/v1.0.3"
CURRENT_STAGE=immutable-release-download
DOWNLOAD=$(/usr/bin/mktemp -d --tmpdir="$STATE_ROOT" .v103-download.XXXXXXXX)
/usr/bin/chmod 700 "$DOWNLOAD"
DOWNLOAD=$(canonical_directory "$DOWNLOAD")
DOWNLOAD_CLEANUP=$DOWNLOAD
DOWNLOAD_CLEANUP_ID=$(/usr/bin/stat -c '%d:%i' -- "$DOWNLOAD")
gh_call release download "$TAG" --repo "$REPOSITORY" --dir "$DOWNLOAD"
[[ "$(/usr/bin/find "$DOWNLOAD" -mindepth 1 -maxdepth 1 -printf '%f\n' | /usr/bin/sort)" == "$EXPECTED_NAMES" ]] || { printf '%s\n' 'error: immutable migration download census differs' >&2; exit 1; }
while IFS= read -r name; do
  CURRENT_STAGE="verify-asset:$name"
  [[ -f "$DOWNLOAD/$name" && ! -L "$DOWNLOAD/$name" && "$(/usr/bin/sha256sum -- "$DOWNLOAD/$name" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/sha256sum -- "$STAGING/$name" | /usr/bin/awk '{print $1}')" ]] || exit 1
  gh_call release verify-asset "$TAG" "$DOWNLOAD/$name" --repo "$REPOSITORY"
done <<<"$EXPECTED_NAMES"
RELEASE=$(release_state) || { printf '%s\n' 'error: immutable v1.0.3 release census changed during download verification' >&2; exit 1; }
verify_release_assets "$RELEASE" || exit 1
[[ "$(/usr/bin/jq -r '.assets[].name' <<<"$RELEASE" | /usr/bin/sort)" == "$EXPECTED_NAMES" ]] || exit 1
if [[ ! -e "$EVIDENCE" && ! -L "$EVIDENCE" ]]; then
  /usr/bin/mv -- "$DOWNLOAD" "$EVIDENCE"
  DOWNLOAD_CLEANUP=
  DOWNLOAD_CLEANUP_ID=
else
  EVIDENCE=$(canonical_directory "$EVIDENCE")
  while IFS= read -r name; do
    [[ -f "$EVIDENCE/$name" && ! -L "$EVIDENCE/$name" && "$(/usr/bin/sha256sum -- "$EVIDENCE/$name" | /usr/bin/awk '{print $1}')" == "$(/usr/bin/sha256sum -- "$DOWNLOAD/$name" | /usr/bin/awk '{print $1}')" ]] || { printf '%s\n' 'error: durable migration evidence differs on resume' >&2; exit 1; }
  done <<<"$EXPECTED_NAMES"
  existing_names=$(/usr/bin/find "$EVIDENCE" -mindepth 1 -maxdepth 1 -printf '%f\n' | /usr/bin/sort)
  [[ "$existing_names" == "$EXPECTED_NAMES" || "$existing_names" == "$expected_with_sidecars" ]] || { printf '%s\n' 'error: durable migration evidence census differs on resume' >&2; exit 1; }
  case "$DOWNLOAD" in "$STATE_ROOT"/.v103-download.*) /usr/bin/rm -rf -- "$DOWNLOAD" ;; *) exit 1 ;; esac
  DOWNLOAD_CLEANUP=
  DOWNLOAD_CLEANUP_ID=
fi
EVIDENCE=$(canonical_directory "$EVIDENCE")

FINAL_CLONE="$TEMPORARY_ROOT/final-target.git"
CURRENT_STAGE=signed-terminal
"${GIT[@]}" clone --mirror --template= "git@github.com:${REPOSITORY}.git" "$FINAL_CLONE" >/dev/null
"${GIT[@]}" --git-dir "$FINAL_CLONE" remote remove origin
TERMINAL="$STATE_ROOT/v103-migration-terminal.json"
TERMINAL_ROOT=$(/usr/bin/mktemp -d --tmpdir="$STATE_ROOT" .v103-terminal.XXXXXXXX)
/usr/bin/chmod 700 "$TERMINAL_ROOT"
TERMINAL_CLEANUP=$TERMINAL_ROOT
TERMINAL_CLEANUP_ID=$(/usr/bin/stat -c '%d:%i' -- "$TERMINAL_ROOT")
TERMINAL_PENDING="$TERMINAL_ROOT/v103-migration-terminal.json"
terminal_args=(create-v103-migration-terminal --intent "$INTENT" --audit "$AUDIT" --audit-sha256 "$AUDIT_SHA256" --receipt "$EVIDENCE/$(/usr/bin/basename -- "$RECEIPT")" --manifest "$EVIDENCE/$(/usr/bin/basename -- "$MANIFEST")" --target-repo "$FINAL_CLONE" --evidence-directory "$EVIDENCE" --gh-binary "$GH_BINARY")
if [[ -f "$STATE_ROOT/v103-migration-incident.json" && ! -L "$STATE_ROOT/v103-migration-incident.json" ]]; then terminal_args+=(--incident "$STATE_ROOT/v103-migration-incident.json"); fi
"${PYTHON[@]}" "${terminal_args[@]}" --output "$TERMINAL_PENDING" >/dev/null
verify_audit || { printf '%s\n' 'error: migration audit changed before terminal publication' >&2; exit 1; }
verify_source_live || { printf '%s\n' 'error: mutable source changed after terminal creation' >&2; exit 1; }
verify_target_controls || exit 1
[[ "$(remote_refs)" == "$EXPECTED_TAG_STATE" ]] || exit 1
verify_staging || exit 1
RELEASE=$(release_state) || { printf '%s\n' 'error: immutable v1.0.3 release census changed before terminal publication' >&2; exit 1; }
[[ "$(/usr/bin/jq -cS '{id,node_id,tag_name,name,body,draft,prerelease,immutable}' <<<"$RELEASE")" == "$(/usr/bin/jq -cnS --argjson id "$RELEASE_ID" --arg node "$RELEASE_NODE_ID" --arg tag "$TAG" --arg title "$TITLE" --arg body "$NOTES" '{id:$id,node_id:$node,tag_name:$tag,name:$title,body:$body,draft:false,prerelease:false,immutable:true}')" ]] || exit 1
verify_release_assets "$RELEASE" || exit 1
[[ "$(/usr/bin/jq -r '.assets[].name' <<<"$RELEASE" | /usr/bin/sort)" == "$EXPECTED_NAMES" ]] || exit 1
[[ "$(/usr/bin/find "$EVIDENCE" -mindepth 1 -maxdepth 1 -printf '%f\n' | /usr/bin/sort)" == "$expected_with_sidecars" ]] || { printf '%s\n' 'error: signed terminal evidence census differs' >&2; exit 1; }
if [[ -e "$TERMINAL" || -L "$TERMINAL" ]]; then
  [[ -f "$TERMINAL" && ! -L "$TERMINAL" ]] && /usr/bin/cmp -s -- "$TERMINAL" "$TERMINAL_PENDING" || { printf '%s\n' 'error: existing migration terminal differs' >&2; exit 1; }
  /usr/bin/rm -- "$TERMINAL_PENDING"
else
  /usr/bin/mv --no-clobber -- "$TERMINAL_PENDING" "$TERMINAL"
  [[ -f "$TERMINAL" && ! -L "$TERMINAL" ]] || exit 1
  /usr/bin/sync -d -- "$TERMINAL" "$STATE_ROOT"
fi
/usr/bin/rmdir -- "$TERMINAL_ROOT"
TERMINAL_CLEANUP=
TERMINAL_CLEANUP_ID=
printf 'v1.0.3 immutable migration PASS: terminal-id=%s prior-authority=%s\n' "$(/usr/bin/jq -er '.terminal_id' "$TERMINAL")" "$(/usr/bin/jq -cS '.prior_release_authority' "$TERMINAL" | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"
