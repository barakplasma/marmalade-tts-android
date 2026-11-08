package app.marmalade.tts.data

import app.marmalade.tts.data.db.VoiceMeta

/** Phonon 31m English q8 model, default Gradium voice. */
object PhononVoiceCatalog {
    const val ENGINE = "phonon-31m-en"
    const val SAMPLE_RATE = 24_000
    const val DEFAULT_VOICE_ID = "$ENGINE:default"
    val voices = listOf(
        VoiceMeta(id = DEFAULT_VOICE_ID, engine = ENGINE, displayName = "Phonon Default",
            languageCode = "en-US", sampleRate = SAMPLE_RATE, gender = null,
            isInstalled = false, sortOrder = 0),
    )
}
