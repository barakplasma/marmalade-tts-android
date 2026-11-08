package app.marmalade.tts.data

import app.marmalade.tts.data.db.VoiceMeta

/** PiVo 1.4M: two synthetic English voices, 16 kHz. */
object PiVoVoiceCatalog {
    const val ENGINE = "pivo-1m-v1"
    const val SAMPLE_RATE = 16_000
    val VOICE_KEYS = listOf("pi", "vo")
    val voices = listOf(
        VoiceMeta(id = "$ENGINE:pi", engine = ENGINE, displayName = "Pi",
            languageCode = "en-US", sampleRate = SAMPLE_RATE, gender = null,
            isInstalled = false, sortOrder = 0),
        VoiceMeta(id = "$ENGINE:vo", engine = ENGINE, displayName = "Vo",
            languageCode = "en-US", sampleRate = SAMPLE_RATE, gender = null,
            isInstalled = false, sortOrder = 1),
    )
}
