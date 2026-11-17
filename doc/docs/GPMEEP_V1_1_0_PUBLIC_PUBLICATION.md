# Publishing the reviewed public 1.1.0 release

The authoritative source release remains `iverydj/gpmeep`. Complete raw
qualification evidence and its private release assets stay there. The public
destination is the existing `iverydj/gpmeep-releases`, repository ID 1372662716
and node ID `R_kgDOUdErvA`. Its v1.0.3 release is a historical source-only release;
it is neither replaced nor described as an immutable binary release.

`scripts/publish-v110-public.py` replaces the former organization migration
controller for this destination. It requires the existing final private release
authority, a privacy-reviewed source snapshot and assets, and an independently
pinned publication audit. The M4, inversion, exact-package, portability and
source-freeze gates remain unchanged.

## Prepare immutable inputs

Finish the ordinary private source release first, using
`scripts/publish-v110-release.sh`. Its final receipt must replay successfully;
its private v1.1.0 tag and published immutable release must match the accepted
source and candidate package. Capture the live private repository/release
metadata and create a source authority with the existing
`v110-release-only-receipt.py create-source-authority` command. That shared
authority path checks the entire private asset manifest, including qualification
and complete-evidence assets. Only its source-authority functionality is reused;
the former target migration and identical-tree snapshot receipts are not.
Run source-authority creation with
`.envs/gpmeep-v1.1-test/bin/python3.11 -I -S scripts/v110-release-only-receipt.py create-source-authority`.
The final-verifier child retains `-I` and loads the contracted dependencies.
Before replay, `verify-v110-final-runtime.py` checks the interpreter path and
SHA-256 and compares the installed explicit package URL/SHA-256 inventory with
the committed contract lock using the pinned Micromamba binary. This uses the
same environment inventory trust level as M0; it does not claim to hash every
installed site-package byte. The private publisher performs this check after
its independent source preflight and before each final receipt verification.
Public preflight also binds every private asset's uploaded state, size and digest
to the full source manifest and verifies each signed asset attestation.

Create a separate public staging repository with exactly these refs:

- `main`: the new reviewed public snapshot, whose single parent is the pinned
  current public `main` commit;
- `v1.0.3`: the unchanged existing public tag object;
- `v1.1.0`: an annotated tag pointing to the new snapshot.

Use an independent `.git` directory, without linked worktrees, local remotes,
includes, transport rewrites or command overrides. The controller permits only
ordinary repository-format, filemode, bare, reflog and author identity settings;
its review binds their digest and execution rechecks it before pushing.
Import only the public repository's history. Copy accepted source files as
content, never by fetching private refs. Apply the permitted omissions and
privacy edits described in `GPMEEP_PUBLIC_SOURCE_VERIFICATION.md`.
`PUBLIC_RELEASE.json` must contain `version: "1.1.0"`,
`binary_source_commit` and `package_sha256` from the original private authority.
The public commit is not the package build commit.

The v1.1.0 annotated tag message is exactly these four lines, with a trailing
newline and the accepted values substituted:

```text
gpmeep v1.1.0
binary-source-commit: FULL_ORIGINAL_BUILD_COMMIT
package-sha256: ORIGINAL_PACKAGE_SHA256
final-receipt-id: PRIVATE_FINAL_RECEIPT_ID
```

Prepare five public assets: the byte-identical accepted package, reviewed
Markdown and HTML reports, a copy of the public metadata JSON, and a SHA-256
file. The checksum file lists exactly the other four assets in filename order,
one `SHA256`, two spaces and filename per line. Reports and metadata must live
outside the private release-asset directory. No private receipts, raw evidence
bundles or arbitrary extra assets are accepted. Release notes must include the
original source commit and package SHA-256 and receive the same privacy review.

## Pinned publication plan

Keep the plan private. Its exact JSON fields are:

| Field | Required value |
| --- | --- |
| `schema` | `gpmeep-v110-publication-plan-v1` |
| `source_repo`, `source_assets` | Absolute private source and release-asset directories |
| `source_authority` | `{ "path": ABSOLUTE_FILE, "sha256": SHA256 }` |
| `final_verifier` | Pinned verifier `path`, `sha256`, full supplemental `arguments`, and `python: {"path": ABSOLUTE_PYTHON3_11, "sha256": SHA256}` |
| `trusted_private_tag_object` | Independently trusted original v1.1.0 annotated tag ID |
| `public_repo` | Absolute isolated public staging repository |
| `public_commit`, `public_tag_object` | Full reviewed commit and annotated tag IDs |
| `prior_refs` | Exactly the current `refs/heads/main` and `refs/tags/v1.0.3` IDs |
| `privacy_policy` | Pinned private-literal/fixture policy file |
| `assets` | Records containing `role`, absolute `path`, `sha256`, `size_bytes` |
| `notes`, `github_cli` | Pinned files, each with `path` and `sha256` |
| `release_title` | `gpmeep v1.1.0` |

Asset roles are exactly `package`, `report_markdown`, `report_html`, `checksums`
and `public_metadata`. The final verifier's fixed receipt, candidate and package
arguments are supplied by the authority replay, not overridden by its argument
list. All files must be regular files without symlink path components.

```sh
.envs/gpmeep-v1.1-test/bin/python3.11 -I -S scripts/publish-v110-public.py \
  --plan /path/to/private-plan.json --plan-sha256 PLAN_SHA256 \
  --offline --output /path/to/new-local-review.json
```

This replays the final authority, source comparison, complete package scan,
public report scan and local history inventory. Omitting `--offline` also checks
the live private/public identities, advertised public refs, original tag and
private release/asset attestations. Both modes are read-only. Neither output
authorizes publication.

## Independent review and execution

An independent reviewer must review the pinned implementation, exact plan and
review report, document content and media, the public remote's advertised refs
and potentially accessible cached/PR objects, and the original package authority.
The externally supplied audit has schema `gpmeep-v110-publication-audit-v1`,
state `GO`, exact `plan_sha256`, `review_report_id`, `implementation` from the
local review, named `reviewer`, and integer zero `critical`, `high` and `medium`
counts under `findings`. Its `checks` object contains exactly four true booleans:

- `documentation_and_media_reviewed`;
- `remote_history_and_cached_objects_reviewed`;
- `exact_package_authority_replayed`;
- `publication_controller_adversarially_reviewed`.

`audit_id` is SHA-256 of canonical JSON without `audit_id`, using sorted keys,
compact separators, UTF-8 and `ensure_ascii=False`. Pass the separately trusted
SHA-256 of the complete audit file. A matching hash is a binding, not a substitute
for independent review; do not self-issue a GO result to bypass that requirement.

```sh
.envs/gpmeep-v1.1-test/bin/python3.11 -I -S scripts/publish-v110-public.py \
  --plan /path/to/private-plan.json --plan-sha256 PLAN_SHA256 \
  --audit /path/to/independent-audit.json --audit-sha256 AUDIT_SHA256 \
  --execute --output /path/to/new-publication-result.json
```

Execution rechecks the live destination, enables immutability for future releases
and disables Actions, atomically pushes only the reviewed main commit and tag,
creates a draft, uploads verified copies of the five accepted assets and checks
the remote hashes before publishing. It then verifies the immutable release and
each asset's signed attestation. GitHub's documented behavior applies:
[immutability affects future releases](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes),
and [release/asset integrity checks](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity)
do not cover GitHub's generated source archives. Their source content is covered
by the separate reviewed Git tree/history inspection.

After an interruption, use the same pinned inputs with `--execute --resume` and
a new output path. Only the exact expected remote refs, draft or already
immutable release and existing asset hashes are accepted. No force push,
tag deletion, asset replacement, package mutation or automatic repair of remote
drift is performed. Investigate any mismatch and obtain a new review when inputs
change. Final numerical/performance and source authority remain prerequisites
even for a resume.
