# Declared scene-linear sequence primaries

M41 declared sequence-primary mechanics are locally verified:806CPU/no skips, no new diagnostics versus baseline or actual M40, independent/main review and actual Qt/owned CPUworker exports. See `benchmark-results/m41-verification-summary.json`. Full colour and model/host acceptance remain unverified. This workflow interprets a declared scene-linear image sequence in a known primary basis. It does not detect colour tags or establish the trained model's colourspace.

`source_colour_profile` defaults to `legacy`. Existing display-coded assumed-sRGB and `frame_dir_linear` workflows retain their numerical path. Two explicit profiles require `frame_dir_linear=true`:

- `linear_bt709_d65`: declared BT.709 primaries and D65 white, with scene-linear RGB; identity conversion bypasses matrix arithmetic.
- `linear_bt2020_d65`: declared BT.2020 primaries and D65 white, converted by a versioned linear 3×3 matrix into the engine's existing assumed BT.709/D65 basis.

The primary chromaticities and D65 white come from [ITU-R BT.709-6, section 1.3](https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en) and [ITU-R BT.2020-2, Table 3](https://www.itu.int/rec/R-REC-BT.2020-2-201510-I/en). The matrix derives from their normalized RGB-to-XYZ bases with a shared white. This operation uses linear light and does not apply either recommendation's nonlinear encoding curve.

```sh
python infer.py --input /local/linear2020-sequence --output_dir /local/fresh-output \
  --frame_dir_linear --source_colour_profile linear_bt2020_d65 --output_format exr
```

The same field is available through shared batch/queue configuration, desktop settings and interactive sequence choices. Explicit declarations require image sequences; video and strict source-witness/OFX binding are currently refused. Legacy choices retain their defaults. The desktop's source preview uses the declared basis, and completed source/render comparisons retain the completed job's interpretation even after editor settings change.

Declared premultiplied sources are unpremultiplied before conversion using the existing coverage/zero policy. RGB alone enters the matrix. Negative and above-one finite values remain available to the existing signed-asinh model input and native Quick Key source-RGB paths. Display/foundation proxies retain their established clipping; alpha remains coverage without an RGB transfer. Invalid or overflowing conversion fails explicitly. Generic image/hint/coverage readers do not acquire an implicit source transform.

Foreground, cutout and checker outputs use the working BT.709/D65 basis: scene-linear for EXR and existing linear cutout choices, otherwise the existing display encoding. There is no automatic conversion back into source BT.2020. Settings/diagnostics and output contracts record the declared source, assumed working basis, transform version and coefficients. Embedded EXR chromaticities and colour-managed host/display interpretation remain unverified. Existing half-EXR finite saturation remains part of the writer contract; select float EXR when the required range exceeds half precision.

Quality references and backgrounds must independently declare the existing `linear_srgb_bt709` reference basis. Binding accepts exact known output contracts, including old reports; it does not silently reinterpret BT.2020 reference media. Source-profile settings and implementation identity participate in conservative recovery/cache identities.

This increment does not provide high-bit video decoding, source transfer/range detection, camera/log/PQ/HLG transforms, chromatic adaptation between different whites, gamut/tone mapping, ICC/OCIO, calibrated displays, pretrained quality validation or managed Resolve colour acceptance. Those requirements remain open.
