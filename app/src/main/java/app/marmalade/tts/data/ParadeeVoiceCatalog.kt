package app.marmalade.tts.data

import app.marmalade.tts.data.db.VoiceMeta

/** Paradee v1.2 shares one 8M-parameter ONNX graph across four Kokoro voices. */
object ParadeeVoiceCatalog {
    const val ENGINE = "paradee-8m-v1_2"
    const val SAMPLE_RATE = 24_000
    val VOICE_KEYS = listOf("af_heart", "bf_isabella", "am_puck", "bm_fable")

    val voices: List<VoiceMeta> = listOf(
        VoiceMeta(id = "$ENGINE:af_heart", engine = ENGINE, displayName = "Heart (US)",
            languageCode = "en-US", sampleRate = SAMPLE_RATE, gender = "female",
            isInstalled = false, sortOrder = 0),
        VoiceMeta(id = "$ENGINE:bf_isabella", engine = ENGINE, displayName = "Isabella (UK)",
            languageCode = "en-GB", sampleRate = SAMPLE_RATE, gender = "female",
            isInstalled = false, sortOrder = 1),
        VoiceMeta(id = "$ENGINE:am_puck", engine = ENGINE, displayName = "Puck (US)",
            languageCode = "en-US", sampleRate = SAMPLE_RATE, gender = "male",
            isInstalled = false, sortOrder = 2),
        VoiceMeta(id = "$ENGINE:bm_fable", engine = ENGINE, displayName = "Fable (UK)",
            languageCode = "en-GB", sampleRate = SAMPLE_RATE, gender = "male",
            isInstalled = false, sortOrder = 3),
    )
}
