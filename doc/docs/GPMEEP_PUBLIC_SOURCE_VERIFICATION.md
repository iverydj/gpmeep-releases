# Public source content verification

`scripts/verify-public-source.py` compares immutable Git commits in the private
build repository and a prepared public repository. It reads Git objects, not
working-tree files, and does not import development history or publish refs.

All retained solver, CUDA, Python runtime, package recipe, build, and validation
tool source must have identical bytes and Git modes. License files cannot be
removed or changed. Only the documented private evidence directories,
development checkpoint, and type-trace database may be omitted.

The permitted content differences are narrow:

- Markdown documentation, attribution, citation, and ignore rules are recorded
  as changes requiring human review. They cannot become executable or symlinked.
- Notebook code and Markdown source cells, metadata, order, and other structure
  remain equal. Only code-cell outputs and execution counts may differ.
- Python test fixtures may replace explicitly listed physical GPU UUIDs with
  distinct synthetic UUIDs. Parsed Python syntax must otherwise remain equal;
  changing an assertion or collapsing two device identities is rejected.
- New files are restricted to documentation and the named public metadata files.
  `PUBLIC_RELEASE.json`, when present, must retain the original binary build
  commit rather than the public snapshot commit.

The external policy contains exactly `forbidden_literals` and
`fixture_replacements`. Keep its actual host identifiers private. The report
includes its digest, not its content. Each known private literal and credential
pattern is checked in file names and contents. Generic home/workspace paths are
counted for review because test fixtures may intentionally contain synthetic
paths. A literal scan does not inspect images or establish complete privacy.

```sh
python scripts/verify-public-source.py \
  --source-repo /path/to/private-source --source-commit FULL_BUILD_COMMIT \
  --public-repo /path/to/public-staging --public-commit FULL_PUBLIC_COMMIT \
  --policy /path/to/private-policy.json --output /path/to/new-report.json
```

The report records SHA-256 for every retained file, the two Git trees, all
omissions, and each allowed difference. Existing reports are never overwritten.

`CONTENT_VERIFIED` is only a source-content result. Public ref/history isolation,
documentation and embedded-media review, package privacy, final numerical and
performance qualification, and exact-package release authority remain required.
This tool cannot authorize a tag, upload, release, or replacement of a measured
package. The legacy identical-tree publication controller must not treat its
report as a legacy mirror receipt.
