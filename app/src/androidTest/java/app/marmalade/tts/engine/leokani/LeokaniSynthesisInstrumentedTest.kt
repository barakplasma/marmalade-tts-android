package app.marmalade.tts.engine.leokani

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import app.marmalade.tts.data.LeokaniVoiceCatalog
import app.marmalade.tts.phonemizer.SharedEspeakData
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/**
 * Runs the real Leokani ONNX graph, eSpeak JNI and Kotlin inverse STFT
 * together on Android. The workflow installs the pinned model beforehand.
 */
@RunWith(AndroidJUnit4::class)
class LeokaniSynthesisInstrumentedTest {
    @Test
    fun offlineEnglishProducesNonSilentAudio() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val engine = LeokaniEngine(context, SharedEspeakData(context))
        try {
            assertTrue("Pinned Leokani ONNX bundle was not installed", engine.isInstalled())
            engine.ensureModelLoaded()
            val wav = engine.synthesize(
                text = "Hello! This is Leokani speaking from an Android emulator.",
                voiceId = LeokaniVoiceCatalog.DEFAULT_VOICE_ID,
                speed = 1f,
            )
            assertEquals(24_000, wav.sampleRate)
            assertTrue("Model generated too little audio: ${wav.pcm.size} samples",
                wav.pcm.size > 8_000)
            val peak = wav.pcm.maxOf { abs(it.toInt()) }
            assertTrue("Model produced silent PCM (peak = $peak)", peak > 100)
        } finally {
            engine.release()
        }
    }
}
