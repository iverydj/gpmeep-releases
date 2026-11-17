# Public release privacy policy

The public repository `iverydj/gpmeep-releases` is the public release channel. Private development, original source authority, and raw
validation evidence remain separate.

Public snapshots include solver/CUDA/Python/build source, license and
attribution, user documentation, checksums, and reviewed report summaries.
They exclude raw logs, local paths, host identifiers, actual GPU UUIDs,
resume checkpoints, type-trace databases, and private development history.

The public snapshot commit may differ from the measured binary build commit
because non-runtime material is omitted and documentation is privacy-cleaned.
Always preserve and publish the original binary build commit and SHA-256
separately. Never claim identical Git trees or substitute the public commit
in the package's immutable provenance. Retained runtime/build source must be
verified byte-for-byte against accepted source; notebook source cells and
non-runtime fixture semantics must remain unchanged.

Conda builds disable `build.include_recipe`: the rendered path-source recipe
contains the private source-staging directory. The reviewed source snapshot
retains the build recipe, while the binary retains its original build commit,
checksums, license, runtime manifest, and package provenance. This prevents the
known recipe-path disclosure at build time; it does not replace the final scan
of all archive members and installed payloads.

The legacy `parentless-identical-git-tree-v1` release-only migration does not
meet this new policy and must not publish to the public target. A replacement
publication adapter requires independent adversarial qualification before
v1.1.0 publication. This does not relax any M4 numerical, timing, package,
portability, or final-receipt gate.

The replacement `scripts/publish-v110-public.py` targets the existing public
repository and preserves its reviewed public ancestry and v1.0.3 source-only
release. The development contract now binds that adapter, the public source
verifier and complete artifact/history inspector. The private source authority
replay is shared with `v110-release-only-receipt.py`; its former target migration
path is not used. See `GPMEEP_V1_1_0_PUBLIC_PUBLICATION.md`.

The legacy `publish-v110-release-only.sh` and
`publish-v103-release-only-migration.sh` still describe the former private,
fresh organization repository. They are retained for legacy evidence replay and
must not be redirected to the live public repository. Independent adversarial
qualification of the replacement and all original release gates are mandatory
before publication; implementing an adapter is not release approval.

Scan every public ref, source archive, report (including embedded media), and
extracted package payload. Removing a branch or rewriting history does not
guarantee old GitHub PR refs or cached commits are inaccessible. Keep visibility
private until those old references are purged or proven inaccessible. No
original private Git refs or unreviewed release assets may be imported later.
