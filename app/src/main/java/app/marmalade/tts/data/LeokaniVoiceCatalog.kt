package app.marmalade.tts.data

import app.marmalade.tts.data.db.VoiceMeta

/** Leokani 7M's English 24 kHz voice. Model weights are CC BY-NC 4.0. */
object LeokaniVoiceCatalog {
    const val ENGINE = "leokani-tts-7m"
    const val SAMPLE_RATE = 24_000
    const val DEFAULT_VOICE_ID = "$ENGINE:default"

    val voices: List<VoiceMeta> = listOf(
        VoiceMeta(
            id = DEFAULT_VOICE_ID,
            engine = ENGINE,
            displayName = "Leokani",
            languageCode = "en-US",
            sampleRate = SAMPLE_RATE,
            gender = "female",
            isInstalled = false,
            sortOrder = 0,
        )
    )
}
