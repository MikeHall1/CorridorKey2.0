# Foreign fully collected package review

EXT-09 extends the schema2 reader without changing the writer layout or original
manifest meaning. This initial foreign all-collected reader/Apply batch is locally accepted; full
product and target-platform acceptance remain incomplete.

The lexical archival codec validates the original shared job/settings while
preserving archived path strings and wire bytes. It constructs no native job and
performs no foreign filesystem access. A single unambiguous origin flavour is
inferred from the original input. Render dependency and archival model paths must
agree with it. Windows drive-absolute and ordinary UNC and single-root POSIX paths
are supported; device/extended namespaces, drive-relative/root-relative paths,
ambiguous separators, dot segments, control characters and unsafe collection
basenames remain explicit errors. Valid literal backslashes/colons in POSIX parent
names retain their native meaning. An unused EXR encoder reference may have its
own origin flavour and is preserved exactly.

Foreign packages require every render dependency collected, including hints,
correction documents, authored/derived tracked masks and guidance resources.
Partially collected/reference-only foreign packages refuse before dependency
probing. Exact native replacements for already collected dependencies still need
explicit verification; this is not general foreign missing-reference relinking.
Original job, model and source strings remain provenance, never implicit local
paths. Collection layout uses producing-flavour basenames/parents; source sequence
order/names/offsets, correction semantic identities, colour/precision and settings
are unchanged. Model/executable references never activate automatically.

Verify authenticates frozen file rows/content. Apply requires explicitly chosen
absolute native output/scratch paths for a foreign archive; a pasted producing-host
path refuses instead of becoming a cwd child. All operational source/hint/resource
paths and correction/temp/output settings become native before shared JobRequest
construction. Native package behavior, including legacy relative destination
resolution, remains available. An absolute foreign encoder for video refuses
without replacement; use an explicitly compatible original configuration and
republish or a suitable EXR workflow. Relative executable paths and bare commands
remain unverified environment references, never portable executables.

Desktop projection uses archival settings rather than native deserialization and
shows archive flavour/scope in its bounded off-thread report and retained Apply
context. It loads draft settings only, preserving queue, model selection and the
resident owner. Foreign publication is deliberately unsupported: first Apply to
an explicit native draft, then create and verify a fresh native collection plan.
Public plan/inspection/pin inputs must be admitted before their paths reach native
filesystem calls; the exact boundary regressions are reviewed under336 and338.

Findings classified before remediation:

- 327: desktop report formatting still constructed a native job from foreign
  provenance; replaced with the path-free codec and visible capability scope.
- 328/329/331: new fixture import/type diagnostics fixed locally without checker
  suppression; 334 adds explicit nonnull hint postconditions after runtime proof.
- 330: preserve valid POSIX literal parent names rather than misclassifying them
  as Windows separators/drives.
- 332: new Qt fixture now uses the physical macOS temp root, retaining production
  alias refusal.
- 333: destination refusal now explains the explicit native absolute requirement.
- 335: reject foreign public ProjectPlan publication before any destination,
  protection or membership scan.
- 336: validate complete effective report/header/pin native associations before
  public inspection/publication filesystem access, with canonical JSON type equality.
- 337: explicit dependency list admission preserves checker narrowing and strict
  malformed-record refusal.
- 338: reread bounded pinned package metadata and compare parsed canonical semantics
  so consistently forged detached manifest/report settings cannot replace the actual
  sealed package provenance. Raw whitespace/length need not match canonical snapshots.

Initial11codec cases pass with no native path resolve/stat/open, strict settings,
class/instance/heterogeneous-set serialization and spelling/encoder tests.
Initial32foreign/native portable/collection cases pass before336, including an
actual moved Windows-flavour fixture with local POSIX payloads, both hint kinds,
schema4 tracked/guidance resources and identical final correction rasters. This
is not a package produced or opened by a native Windows application. Two new actualQt cases PASS0.905s and15existingnativeQt cases PASS2.099s on
pre338 source, touchedstatic0; final full gate covers the final admission fix.
Final independent substantial review passed41codec/foreign/native portable CPU
cases0.638s and2actualQt cases0.926s, and replayed338 tampering now refusing with
unchanged package/empty scratch/no outputs. No unresolved local blocker. Final frozen **1408CPU tests PASS103.288s/no skips**;401Python files/full compilation/import/dependency/static regression/whitespace gate PASS,0new vsbaseline AND actualc357cad.

No dependency/environment/download, engine/model/precision/colour/default change,
asset publication or migration of unrelated projects. Native Windows filesystem,
packaging, CUDA, representative quality and Resolve acceptance remain NOT VERIFIED.


Main distinct self-review against AGENTS.md and both briefs: inspect → budgeted
Verify → deliberate Apply remains connected, one owned UI I/O operation, no model
activation/queue submission, no foreign dependency probes, exact correction
identity/rasters and unchanged original bytes. Dataclass class/instance admission,
caller-replaced records, heterogeneous sets and bool/integer canonical JSON
contracts were inspected explicitly and exercised by the focused tests. Existing
relative native destinations remain compatible. New foreign lexical limitations
produce explicit errors rather than silently interpreting archived strings.

Six-file reviewed source/test mapping:
`00294cd9d2f5786145439dc58edad84f71003578dfb01470d269398d7a264a21`.
Final35portable/collection cases PASS0.638s/-Werror;11lexical cases PASS0.009s,
focusedRuff/Pyright0. Final frozen repository source identity `e61fd11e399a0a631dca193168c28d2ecaf398271f351526235f65a1cfa722cc`, unchanged through the complete gate. Prior unchanged-driver3x1137CPU headless benchmark reused only for applicable orchestration scope, with no inference speed/quality claim.
