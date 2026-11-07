# Leokani 7M prototype (draft, not production-ready)

This branch adds a native Kotlin/ONNX inference engine for
[Leokani 7M](https://huggingface.co/milliwattlabs/leokani-tts-7M).

It uses the ONNX Runtime already present in Marmalade, the app's eSpeak-ng
JNI wrapper, and a pure-Kotlin inverse STFT. No Python interpreter is needed
inside Android.

## Current status

- Native engine and voice catalog are wired into EngineRegistry.
- ONNX contract: `ids` int64 [1, T], `speed` float32 [1] -> `mag` and
  `phase` float32 [1, 513, frames].
- 24 kHz mono PCM16 output; speed adjustment is handled downstream.
- The iSTFT implementation has been compared against the upstream NumPy
  reconstruction for synthetic inputs.
- Android app build / on-device listening have **not** been verified.
- **Not yet installable in the Engines tab.** No downloadable tar.gz with
  a pinned SHA256 is hosted yet, so there is intentionally no
  EngineCatalog entry or EngineInstaller verification case.

## Packaging the model

```sh
python -m pip install huggingface_hub
python tools/leokani/make_bundle.py
cat leokani-bundle-metadata.json
```

The archive contains the upstream int8 ONNX model and its config.json.
Once the archive is uploaded to a stable HTTPS location, add an
`EngineDescriptor` to `EngineCatalog.kt` with its actual size and SHA256
and add a layout case to `EngineInstaller.verifyLayout`. Do not use
placeholder hashes or links: the installer validates them.

For manual debug testing, extract the archive into
`filesDir/engines/leokani-tts-7m/` in the installed app, then
launch Marmalade and choose the Leokani voice. Some voice-installation
state may require further UI integration.

## Remaining PR work

- Build against the real Android project and fix compilation issues.
- Provide a stable model archive and register it in the installer catalog.
- Test real text, pronunciation normalization, clause chunking, and
  end-to-end generated PCM against upstream.
- Benchmark cold start and real-time factor on Android hardware.
- Add model credit in the voice selection UI, plus any required notices.
- Have the maintainer set the quality/speed grades based on device tests.

Model weights: CC BY-NC 4.0; this branch does not contain weights.
