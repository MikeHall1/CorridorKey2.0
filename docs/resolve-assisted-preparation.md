# Resolve-assisted preparation: supported-interface investigation

EXT-10 is in progress; **no host acceptance or automatic attachment is claimed**.
The current effect serves prepared frames; neural inference remains in the shared
Python engine. The inspected adapter exposes `CacheFile` and `CacheStart`.
Existing `CachePreparation` supports cancellable sealing and progress; the worker
supports rendering/progress/cancellation. Neither provides editor selection.

The local machine has no Resolve application or installed Developer/Scripting
tree at the normal macOS installation locations inspected. Windows/Resolve/GPU
execution is NOT VERIFIED. The vendored official OpenFX headers provide real
local interface evidence, independently of the native fixture host.

Official interface findings:

| Operation | Supported contract and limitation |
|---|---|
| Clip/range identification | OFX provides the effect instance's Source clip and effect-time ranges. The optional timeline suite exposes time/bounds, not an original-media filename or editor selection. [Timeline suite](https://openfx.readthedocs.io/en/latest/Reference/api/struct/structOfxTimeLineSuiteV1.html) |
| Capture upstream pixels | `clipGetImage` returns Source images; release image handles before the originating action returns. Copy pixels while leased before handing independent data to a companion. [Image suite](https://openfx.readthedocs.io/en/main/Reference/suites/ofxImageEffectSuiteReference.html), [thread/recursion contracts](https://openfx.readthedocs.io/en/main/Reference/ofxThreadSafety.html) |
| Analysis progress/cancellation | Optional OFX progress supports cancellable modal analysis during `InstanceChanged`; this does not establish a general background completion callback. [Progress suite](https://openfx.readthedocs.io/en/main/Reference/suites/ofxProgressSuiteReference.html) |
| Attach/refresh | Set parameters from permitted host actions, including `InstanceChanged`/interact, with appropriate invalidation declarations. Do not call host parameter setters from a companion completion thread. [Parameter suite](https://openfx.readthedocs.io/en/main/Reference/suites/ofxParametersSuiteReference.html) |
| Resolve scripting | Official support release text inspected on2026-10-23 reports selected timeline clip scripting in Resolve21.0.4. This is version-dependent evidence, not a verified signature or source-offset/OFX attachment contract. Inspect the target installation's Developer docs. [Blackmagic support](https://www.blackmagicdesign.com/support) |
| Fusion tool scripting | `Composition.GetToolList(true)`, input enumeration and `Operator.SetInput` are documented for Fusion tools; this does not prove the installed Resolve exposes this OFX instance under assumed IDs. Discover actual inputs and test target behavior. [Official Fusion scripting guide](https://documents.blackmagicdesign.com/UserManuals/Fusion8_Scripting_Guide.pdf) |

The following minimal workflow is an **implementation proposal inferred from
those interfaces**, not an implemented host workflow:

1. Select this effect instance and explicitly confirm integer host In/Out,
   native dimensions, working colour space and a fresh local capture destination.
2. In a permitted host action, copy supported upstream pixels into an owned
   bounded local sequence; honor progress, cancellation and image lifetimes.
   Reject unsupported time/colour/field/resolution cases before submission.
3. Submit the captured sequence through the existing shared worker and seal
   completed exports. Never run neural models in Resolve render callbacks.
4. Validate the fresh completed cache and mapping, then show its `CacheFile` and
   `CacheStart` for explicit attachment. Preserve the previous selection if any
   stage fails. A later explicit host action may attach validated parameters.

This processes pixels at the effect location. Original-camera media, proxies,
upstream transformations and host timeline identity must have separate provenance.
The existing CKOFX002 source-bound reader checks its selected upstream range and
refuses retiming interpolation, nonnative scale, fields, incompatible geometry/
FPS and pixel mismatches. Those conservative checks must survive preparation.

Next prerequisite: target Resolve version/edition and installed official Developer
API docs, followed by a Windows x64 build and actual host capture/progress/abort/
persistence tests. Final acceptance also needs CUDA, compatible weights, local
permitted footage, known host colour settings and viewer/export/retime/proxy/
multi-instance/previous-cache preservation scenarios. SDK/native mocks cannot
replace these runs.

Independent read-only investigation inspected Git blobs:
`adapter.cpp` `0f608ecea097725ee494db549b135022d70251e7`,
`ofx_cache.py` `42c9b53c2308d7669b2ba5b52b5b2145e1737644`,
`ofx_cache_task.py` `44d78641aab164926e88fcb3df580ab264fa7b4f`.
No host calls, installs or asset downloads were performed.
