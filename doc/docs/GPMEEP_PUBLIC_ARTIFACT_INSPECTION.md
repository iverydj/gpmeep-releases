# Public artifact inspection

`scripts/verify-public-artifacts.py` complements the public-source equivalence
check. It reads every member of a Conda v2 package, including metadata, tar
headers and link targets, without extracting files or modifying the archive.
Rendered `info/recipe/` content is rejected. The original package SHA-256 is
recorded and checked again after inspection.

The inspector checks known private literals and Slack/GitHub credential
patterns in files and names. It also checks JSON Unicode escapes, common
HTML/URL escapes, and base64 data URLs in report text. Physical GPU UUIDs are
rejected; explicitly mapped synthetic fixtures, zero-prefixed fixtures, and
existing repeated-digit and ascending/descending hexadecimal test fixtures are
allowed. Generic sample filesystem
paths are inventoried for review, rather than silently declared private or safe.
The same private policy file is used by `verify-public-source.py`.

With `--public-repo` and `--private-repo`, every local public ref, reachable
commit, tag message, tree and file version is inspected. Deleted content in an
ancestor remains in scope. Commits shared with private development history,
unexpected refs, alternate object stores, grafts and shallow history are
rejected. This local inventory does not prove that remote PR refs or cached
GitHub objects are absent. A publisher must compare a fresh remote inventory
and obtain the separate history review before uploading anything.

```sh
python scripts/verify-public-artifacts.py \
  --policy /path/to/private-policy.json \
  --package /path/to/accepted-candidate.conda \
  --asset /path/to/reviewed-report.html \
  --public-repo /path/to/public-staging \
  --private-repo /path/to/private-source \
  --output /path/to/new-inspection.json
```

Package, additional assets and local history can also be inspected separately.
All reads are bounded. Existing reports are never overwritten. The result pins
the implementation, privacy policy and inspected bytes, and records embedded
media hashes for review.

`LITERALS_CHECKED` never authorizes publication. A literal scanner cannot read
text rendered inside a picture, judge whether prose reveals private facts, or
interpret arbitrary encoded executable content. Independent documentation and
media review, fresh remote-ref inspection, complete numerical/performance and
portability qualification, the final exact-package receipt, and an independent
publication audit remain mandatory. Do not upload this private inspection
report automatically: it may inventory private evidence or rejected paths.
