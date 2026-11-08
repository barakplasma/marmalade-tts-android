package app.marmalade.tts.engine.leokani

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import app.marmalade.tts.data.LeokaniVoiceCatalog
import app.marmalade.tts.install.EngineFilesDir
import app.marmalade.tts.install.EngineInstaller
import app.marmalade.tts.install.InstallState
import app.marmalade.tts.install.NativeEngineHandle
import app.marmalade.tts.install.UrlHttpFetcher
import app.marmalade.tts.phonemizer.SharedEspeakData
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/**
 * Integration test on a real Android runtime: download the published model
 * via Marmalade's production installer, verify its SHA256/layout, run ONNX,
 * phonemize with the APK's eSpeak JNI, then save non-silent PCM for listening.
 */
@RunWith(AndroidJUnit4::class)
class LeokaniSynthesisInstrumentedTest {
    @Test
    fun installsRealBundleAndProducesEnglishSpeech() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = LeokaniVoiceCatalog.ENGINE
        File(context.filesDir, "engines/$name").deleteRecursively()
        val installer = EngineInstaller(
            filesDir = EngineFilesDir { context.filesDir },
            engineHandle = NativeEngineHandle { },
            httpFetcher = UrlHttpFetcher,
        )
        val result = installer.install(name) { }
        assertTrue("Engine download or SHA256 verification failed: ${result.exceptionOrNull()}",
            result.isSuccess)
        assertTrue("Installed Leokani assets not recognized by verifier",
            installer.verify(name) is InstallState.Installed)

        val engine = LeokaniEngine(context, SharedEspeakData(context))
        try {
            assertTrue("Engine did not recognize downloaded assets", engine.isInstalled())
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
            val pcmBytes = ByteBuffer.allocate(wav.pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in wav.pcm) pcmBytes.putShort(sample)
            File(context.filesDir, "leokani-smoke.pcm").writeBytes(pcmBytes.array())
            println("LEOKANI_E2E_OK samples=${wav.pcm.size} peak=$peak")
        } finally {
            engine.release()
        }
    }
}
