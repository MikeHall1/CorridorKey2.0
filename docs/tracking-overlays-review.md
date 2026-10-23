# Spatial tracking diagnostics review

Ninth extension batch, 2026-10-23. Main distinct integration self-review and
independent core, adapter and actual-UI reviews accept the documented local scope;
the final frozen CPU/static gate also passes for this unchanged reviewed state.

Reviewed390-file Python mapping:
`c7ef2ce7e733edfa9b5184f2450bb50275448ee6271b77218097a5b0c690de24`.
Exact mapping/logs: `benchmark-results/ext07-connected-final/`.
Parent:865529eafa362bd8e42305ae10c2ab325d1a4de0. Initial failed fixture evidence:
`benchmark-results/ext07-integration-initial/`.

Main review traced actual flow candidate maps into immutable bounded proxies,
receipt publication, source/frame/document/decode/candidate/native-proxy binding,
exclusive leaf ownership, display-only RGBA composition and the editor service.
It inspected default-off capture, separate mask/map budgets, no extra flow pass,
no-map anchor/blocked semantics and first rejected-mask diagnostic support.
Map inspection never changes the tracked-mask inspected flag. Source/decode
verification runs before/after display without modifying source bytes or colour
semantics. The full source read is deliberate and cancellable, not a cheap or
continuous background check.

The UI review followed dirty-draft guards for inspect/hide/navigation/range,
local-to-source offsets, read-only canvas and disabled subject-click mode,
explicit real-source reload before new correction, post-edit map invalidation,
and late completed diagnostic display suppression after cancellation. Diagnostic
maps are never interpreted as Keep/Remove/matte constraints. Accept commits masks
before map-only cleanup; failures retain private subdirectory warnings/ownership
without deleting accepted assets or misreporting committed edits.

Independent review accepts exact core497ab34f23ef8ea86dd03a9fb1cfbded99b93fcee7d705db3bc865645d6c696a
and adapterb4b9efefcf509a728548df640b0744b96fc47ad36206249c9618f475e1270b56.
Core12tests PASS0.045s and independently12PASS0.028s; adapter8new plus kernel/
tracking integration33tests PASS and independently33PASS0.948s;5actualQt tests
PASS1.732s and independently new+existing17QtPASS4.149s. Focused Ruff/Pyright
are clean. The new late-cancel fixture initially used a nonexistent model class;
issue315 records correction to actual CorrectionEditor and explicit optional
snapshot/draft assertions, preserving source/native-pixel/history checks. No
production diagnostic suppression or checker/baseline changes occurred.

Main and independent reviewers explicitly inspected dataclass instance/class
contracts, strict bounded primitive JSON and heterogeneous-set refusal in new
metadata. Existing diagnostics.json_value retains deterministic heterogeneous-set
serialization and instance-only asdict; its8 regressions remain in the full gate.
No inference, quality, colour, precision, model/dependency or persisted correction
format defaults changed. No models/footage were downloaded or environment altered.

Limits: initial tracking-derived uncalibrated maps only, quantized spatial
proxies, no authored-anchor confidence, no invented scalar-to-spatial evidence,
no proven occlusion detector or saved-map reconstruction. Cuts/resets still
restrict tracking; stopped intervals without actual maps remain unavailable.
Actual native-RAM bounds and hostile filesystem mutation guarantees are not
claimed. Physical output data/source files remain untouched by map inspection.
Representative false positives/missed defects, pretrained GPU quality/performance,
Windows and Resolve acceptance remain NOT VERIFIED. Existing static005/nativeQt126
remain open; these passes do not resolve their outstanding scope.

Final gate: **1348CPU tests PASS91.069s/no skips**,390Python files; complete
compilation/import/requirements/VCS/dependency/whitespace/static checks PASS.
Zero new Ruff/Pyright fingerprints vs unchanged baseline AND actual865529e.
Existing519Ruff/684Pyright errors17warnings remain, not static-clean. All source
hashes stayed unchanged. Prior unchanged-driver3x1137CPU headless benchmark is
reused for applicable orchestration scope only; new overlay source has its own
full suite, with no new inference-performance claim.
