# Colour, precision and optional refinement

The current working contract assumes sRGB/BT.709 RGB primaries. Display-coded input uses the inherited sRGB transfer; `--frame_dir_linear` explicitly selects scene-linear frame input. There is no automatic ICC, OCIO, PQ/HLG, gamut/primaries conversion or camera-log interpretation. Convert other working spaces deliberately before using this path. EXR output alone does not establish correct colour management or host interpretation.

## Input and model boundaries

Frame directories accept the existing PNG/JPEG/EXR discovery rules. Decoded uint8 and uint16 RGB/grayscale images normalize by 255 and 65535 respectively into float32. Float EXR remains float32, with signed/above-one values retained in explicit scene-linear mode. Adjacent uint16 codes remain distinct through CPU decode and the float32 source-RGB draft path. Mixed decoded dtypes are accumulated in `decoded_source_dtypes_observed`; the field describes OpenCV's decoded arrays, not original EXR half/float channel metadata. No extra full-clip precision scan is performed.

The neural/foundation input still converts to bfloat16, as before. RVM/batons still use float16 caches; IID feature caches use float32. This is not an end-to-end sixteen-bit neural precision claim. Model RGB is signed asinh scene-linear with the existing reference/knee; the reference value is not an HDR clamp. Foundations/motion/refinement use explicit SDR proxies, separate from scene-linear RGB. Negative/above-one source values can survive float EXR export while proxies/previews clip them as documented.

OpenCV video remains a **legacy uint8 display path** with unverified source bit depth/HDR transfer, now stated in probe output and diagnostics. Unexpected non-uint8 decoded video arrays are rejected. Production high-bit/PQ/HLG video, validated colour transforms and variable-rate timestamp handling remain unfinished; an explicit validated scene-linear image sequence is the available HDR route. No GPU inference fallback is introduced on this Mac.

Image/hint NaN and Inf values, geometry changes within a frame sequence, unsupported storage types, float32 model-encoding overflow, invalid model-output shapes/values and decoded foreground overflow fail explicitly. EXR writer submissions must be finite. Preview clipping does not silently repair a requested foreground that decoded to Inf. Error cleanup preserves partial-output status and closes owned resources; current native/GPU calls cannot be killed cooperatively.

## Alpha association

Alpha is coverage and receives no RGB transfer function.

- `--frame_dir_alpha ignore` (default) retains legacy RGB while discarding an input alpha channel. It does not infer association from file extension.
- `--frame_dir_alpha premultiplied` requires explicit scene-linear RGBA frame-directory input. Native RGB divides by coverage where alpha >1e-6; RGB becomes zero at/below that floor, without adding a denominator bias at alpha=1. Missing/invalid alpha or overflow fails. Display-coded/video unpremultiplication is rejected because its association space is not established.
- `--cutout_alpha_mode straight` is the unchanged default.
- `--cutout_alpha_mode premultiplied` requires an EXR cutout output. Despilled scene-linear RGB is multiplied by the final coverage; filenames include `_premultiplied`. This does not alter `fg` output. Premultiplied video is not exposed.

For linear straight RGB F, background B and coverage a, composites use `F*a + B*(1-a)`. Premultiplied cutout stores P=`F*a`, so the equivalent composite is `P + B*(1-a)`. CPU fixtures compare these over black/white/grey/saturated solids and generated textured backgrounds, including alpha 0/1/soft, signed and HDR RGB. Actual photographic/held-out comparison remains unavailable.

Output/job contracts and settings logs identify representation, assumed primaries, association and precision. Alpha-only output is labelled data, with no RGB primaries/transfer assigned to its pixel values. Legacy alpha preview movies retain existing RGB container tags; consume their pixel values as coverage data and bypass RGB colour transforms. OpenCV-written EXR chromaticities/association metadata and host import interpretation are not certified here; use the declared job/settings contract and validate in the target host.

Float EXR retains finite signed HDR RGB. Half EXR retains the inherited finite saturation outside +/-65504 before half conversion; this policy is now explicit in output contracts. Preview RGB clips to [0,1] and uses the existing uint16 encoder input. Neither changing bit depth nor making a sharper matte implies higher accuracy.

## Spatial alpha adjustment

`--alpha_refine_radius` defaults to **0/off** and accepts integer 0..4. When enabled, `--alpha_refine_strength` is in [0,1] (default 1), and `--alpha_refine_sigma` is in [1e-6,1] (default 0.1). The sigma lower bound prevents float32 denominator underflow; it is not silently clamped.

For each partially covered centre pixel only, the CPU operator averages neighbouring alpha using a Gaussian spatial factor and an RGB-proxy-distance factor:

`weight = exp(-RGB_distance_squared/(2*sigma^2)) * exp(-offset_squared/(2*spatial_sigma^2))`

`spatial_sigma=max(0.5,radius/2)`. Borders replicate edge samples. Strength blends the weighted result with the input; exact 0/1 centres remain fixed. No temporal smoothing, learned confidence, trimap or new model is claimed. This operation can remove fine partial coverage or motion blur; accuracy requires reference/visual evaluation. It is opt-in and may be slower.

Neural frames use a current-source proxy derived from the inherited model-input representation; non-anchor draft frames use the current native float32 source proxy. Each neural/output frame is evaluated once. Refined anchor alpha may guide propagation. Authored corrections are applied afterward and remain authoritative, with raw neural/refined alpha separated from propagation conditioning and final ordered constraints. Radius/strength zero keeps the adjustment off. Diagnostics report evaluated/changed frames, parameters and CPU filter wall time; exact numerical changes can include rounding differences. These counters are not accuracy or speedup measurements.

## Screen-aware despill

The defaults retain existing green despill, strength 0.5 and `--despill_space legacy`. The inherited path works in linear RGB for EXR/linear cutouts and in display-coded sRGB for ordinary preview cutouts/checkers. Default CPU tests match the original equations exactly. `fg` remains the raw reconstructed (or explicit draft-source) foreground; despill applies to cutout/checker.

New opt-in controls have working effects:

- `--despill_screen green|blue` chooses the excess channel relative to the mean of the other two. Removed excess is distributed equally to those channels.
- `--despill_space linear` performs suppression in linear RGB, including preview composition before transfer encoding. Legacy space remains available to preserve previous appearance.
- `--despill_preserve_luma` requires explicit linear space. It adds a neutral correction to preserve weighted linear Y with coefficients (0.2126,0.7152,0.0722), before output clipping/half saturation. It can produce signed RGB and is valid only under the stated primaries assumption.
- `--despill_edges_only` scales strength by `4*a*(1-a)`, preserving alpha-0/1 colour from this operation and concentrating it on soft coverage. It is not an inferred edge-confidence mask.

Job/output contracts record the selected screen/space/association. Linear-Y preservation is not promised after clipping, encoding or importing through a different colour space. Reference footage, green/blue screens, transparency, hair/blur and protected foreground colours still need actual visual/metric comparisons.

## Validation and remaining work

CPU tests cover native uint16 levels, mixed decoded dtype reporting, invalid/geometry failures, scene-linear association, composite equivalence, legacy despill parity, blue-screen/linear-Y/edge math, spatial adjustment effects/endpoints/cancellation, correction precedence, exactly-once cross-path evaluation and model overflow cleanup. Runtime neural modules are deterministic substitutes. Real GPU precision/parity/performance, target-host EXR/movie metadata, photographic composites and production colour transforms remain unvalidated.

Next: reference quality metrics and comparisons, then higher-precision video/managed source-colour paths and remaining host validation. Keep those explicit pending items separate from locally verified image/math behavior.
