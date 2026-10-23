# Run checkpoint — milestone27, 2026-10-22

Five implementation/verification cycles completed. Previous goal turn: **progress**, revalidated milestone26 packaging/launch work and fresh486-test/218-file baseline before edits. Milestone26 is archived in `checkpoints/milestone26.md` and `.ralph/checkpoints/milestone26.json`. Full development-brief scope remains active/incomplete; no competing supervisor launched.

## Completed cycles

1. Retrieved8 unmodified official OpenFX interface headers plus BSD3 licence through the connected GitHub read capability, pinned commit `c3f1c2dcf4f51762b794c4fc21181b4a501be1bf`. Verified Git blob identities; preserved notices/provenance. Added completed linear foreground/alpha EXR sealer and bounded SHA256-checked C++ frame reader. Fixed inspected multiplication overflow/indexing and reproduced /var path alias lookup. Actual runtime-fixture exports pass native random/corrupt/table/overflow tests.487 tests/full gate pass.
2. Implemented experimental native filter entry/descriptor/instances/parameters/render against real ABI headers, plus CMake target recipe and actual shared-module enumeration test. Explicit CacheFile/CacheStart controls; conservative render serialization; unsupported inputs fail. Local Clang C++17 strict compile and488-test gate pass.
3. Added minimal official-suite host harness exercising adapter callbacks, image leasing, two separately created instance setups, random-order reads, signed/padded row strides, premultiplied output and refusal/abort. Actual returned image field/PAR/association is validated rather than assumed from preferences. Native pixels independently match actual exported data.488 tests/full gate pass. This is not actual concurrent Resolve instances.
4. Added source-RGB draft filename labels and shared-record schema1 admission. Independent nonuniform signed-HDR fixture checks orientation/channels/coverage and cropped-window byte sentinels under address/undefined sanitizers; short strides/field/PAR/association refusals release images. Fixed fixture's RGB/BGR writer boundary, retaining its independent RGB expectation.489 tests/full gate pass.
5. Added companion frozen-app dispatch route, OpenFX licence/provenance retention in desktop data and Windows x64 native bundle/resource layout. Corrected multi-config output directory and rejected non-Intel-x64 Windows compilation from primary packaging/CMake semantics. Generated actual shared-runtime CPU-fixture exports → sealed cache → sanitized native host output, max absolute pixel error0 and original EXR hashes unchanged. Final489-test gate passes; repeated CPU benchmark recorded below.

## Verification

- Fresh baseline: `benchmark-results/m27-start-verification.txt`, archived `m27-start-baseline/`.
- Per-cycle archives `m27-cycle1/` through `m27-cycle4/`; final `m27-final/`, log `m27-final-verification.txt`. Initial final gate before bundle layout correction retained in `m27-final-before-bundle-layout/`.
- `.ralph/runs/manual-check/verification.json`: success true, all4 commands return0. **489 tests PASS/no skips;221 Python files,0 syntax errors.** Runtime imports, requirements/VCS identity, installed dependency consistency, compilation and final whitespace pass.
- Ruff520 findings; Pyright696 errors/17 warnings (713 diagnostics), **0 new fingerprints** against unchanged baseline. Not static-clean. New/touched Python interop/dispatch/bundle tests focused Ruff/Pyright0.
- Owned native cache/adapter/harness compiled with Clang21/C++17 `-Wall -Wextra -Werror -pedantic`; actual shared-module export enumeration and official-suite callback harness pass. Address/undefined sanitizer checks pass with independent nonuniform signed-HDR/row/window expectations. Native byte/pixel tests are not actual Windows/Resolve/GPU acceptance.
- Header and licence Git blob identities match pinned official source. Existing dataclass instance/class and mixed-set serializer contracts inspected/retained. No checker weakening, suppression, baseline refresh, package install, user environment upgrade, model/footage download or publication/upload. Pylance editor diagnostics NOT VERIFIED.
- CPU benchmark: see final line below. These are correctness-suite elapsed times, not inference/cached-playback FPS or quality measurements.

## Actual generated workflow

`benchmark-results/m27-generated-summary.json`, script `m27-demo.py`, final log `m27-generated-demo-final.txt`:

- Real shared runtime CPU decode/temporal/export orchestration with **explicit neural substitutes**, not pretrained inference.
- Source start2/count4, actual completed native EXR foreground/alpha → companion seal → C++ official-suite host harness with address/undefined sanitizers.
- Two sequentially created instance setups, explicit host-time offsets and positive/negative padded strides. Random requests produce native premultiplied pixels matching actual exports: **max absolute error0**. Original EXR content hashes unchanged.
- GPU FPS null; Resolve host and neural quality NOT VERIFIED. Artifacts remain local. No effect/SDK installation into a system host/plugin directory.

## Contracts and issues

[ofx-cache.md](ofx-cache.md) defines file/schema/bounds/digests/mode/association/time/host limitations and target commands; [resolve.md](resolve.md) records SDK investigation versus actual implementation; [packaging.md](packaging.md) retains separate desktop acceptance.

HEALTH-ENV-152 documents missing CMake and shell DNS; pinned small official headers were available through connected GitHub reads.153–158 were classified before corrections (bounds/index lookup, path alias, actual image contract, draft/schema, fixture BGR boundary and native bundle layout). Failed focused evidence retained in m27-cycle1-focused and m27-cycle4-focused logs.

**HEALTH-OFX-159 remains open:** the new shared-module ABI test loads its library in the parent unittest process while the temporary directory cleanup occurs before the local library variable leaves scope. Local Mac tests pass; Windows loaded-module/file lifetime is unverified. Next isolate enumeration in an owned subprocess that exits before cleanup. This is deferred at the five-cycle boundary, not declared fixed. Existing native Qt mechanism126 and broad static backlog remain open.

## Material limits and full remaining scope

The native effect is an **experimental explicitly selected prepared-cache filter**, not live host keying or complete Resolve integration. It does not authenticate that a same-geometry/FPS host source is the original producer, automatically analyse/recompute/invalidate host shots, interpolate retimes, support proxy/scaled/fielded/non-square output or implement managed host primaries/transfers. Cache/time/colour selection is explicit; missing/incompatible/corrupt data fails. Stronger source-content/time/colour binding and host invalidation remain required before product acceptance. Row-level abort checks cannot interrupt all file/hash work immediately.

File/header/frame checksums are corruption checks, not authentication of a replaced whole file or path-only producer metadata. Max16Mi pixels/256MiB payload and1,000,000-frame/32MB table admission are not total host/process/disk budgets; conversion can hold additional arrays and cache creation duplicates output storage. No retention deletion/quota/power-loss durability claim. Failed sealing removes only its owned stage; private partial directories may remain.

Actual current prerequisites: `m27-prerequisites.json` reports CUDA unavailable, all5 configured model/foundation asset paths absent, CMake/FFmpeg absent, Clang available. Direct local diagnostic native builds do not establish Apple inference/product support. CMake execution, MSVC/Windows module loading, real Resolve/Fusion controls/lifetime/concurrent instances/colour/export, real assets/held-out footage and pretrained GPU quality/performance remain **NOT VERIFIED**. Licensing/distribution, packaged clean-machine behavior and remaining desktop/engine acceptance stay open.

Next: isolate native ABI loading159; provide usable desktop cache preparation and stronger source/time/colour binding/invalidation; inspect target/upstream SDK/plugin behavior and test actual Windows Resolve. Continue independent total budgets, retention/portable backups, advanced corrections, high-bit/managed colour, direct-seek/Stage2/writer recovery, controlled alternating/stress/held-out evaluation, packaging and the rest of the full brief. Goal is neither complete nor blocked.

Machine checkpoint `.ralph/checkpoint.json`; tasks `.ralph/plan.json`; issues `.ralph/issues.json`.
Actual CPU benchmark3 repetitions complete/success: median35.486684291041456s, min35.330594875034876s, max35.92191262508277s. All489-test repetitions pass. These are correctness-suite elapsed times, not inference or cached-playback FPS, GPU speedup or quality evidence.
