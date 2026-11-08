package app.marmalade.tts.engine.phonon

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import app.marmalade.tts.data.PhononVoiceCatalog
import app.marmalade.tts.install.EngineFilesDir
import app.marmalade.tts.install.EngineInstaller
import app.marmalade.tts.install.InstallState
import app.marmalade.tts.install.NativeEngineHandle
import app.marmalade.tts.install.UrlHttpFetcher
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/** Real Phonon native ARM64/x86_64 FFI + model download + PCM validation. */
@RunWith(AndroidJUnit4::class)
class PhononSynthesisInstrumentedTest {
    @Test fun installsModelAndSynthesizesEnglishAudio() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = PhononVoiceCatalog.ENGINE
        File(context.filesDir, "engines/$name").deleteRecursively()
        val installer = EngineInstaller(
            filesDir = EngineFilesDir { context.filesDir },
            engineHandle = NativeEngineHandle { },
            httpFetcher = UrlHttpFetcher,
        )
        val installed = installer.install(name) { }
        assertTrue("Phonon download failed: ${installed.exceptionOrNull()}", installed.isSuccess)
        assertTrue("Phonon layout failed", installer.verify(name) is InstallState.Installed)
        val engine = PhononEngine(context)
        try {
            engine.ensureModelLoaded()
            val wav = engine.synthesize(
                text = "Hello. This is Phonon speaking from an Android emulator.",
                voiceId = PhononVoiceCatalog.DEFAULT_VOICE_ID,
                speed = 1f,
            )
            assertEquals(24_000, wav.sampleRate)
            assertTrue("Phonon audio too short", wav.pcm.size > 8_000)
            val peak = wav.pcm.maxOf { abs(it.toInt()) }
            assertTrue("Silent native Phonon PCM", peak > 100)
            val bytes = ByteBuffer.allocate(wav.pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (sample in wav.pcm) bytes.putShort(sample)
            File(context.filesDir, "phonon-smoke.pcm").writeBytes(bytes.array())
            println("PHONON_OK samples=${wav.pcm.size} peak=$peak")
        } finally {
            engine.release()
        }
    }
}
