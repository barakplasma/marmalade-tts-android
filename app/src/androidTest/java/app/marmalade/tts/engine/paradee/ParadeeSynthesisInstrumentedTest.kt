package app.marmalade.tts.engine.paradee

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import app.marmalade.tts.data.ParadeeVoiceCatalog
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

@RunWith(AndroidJUnit4::class)
class ParadeeSynthesisInstrumentedTest {
    @Test fun installsModelAndSynthesizesFourVoices() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = ParadeeVoiceCatalog.ENGINE
        File(context.filesDir, "engines/$name").deleteRecursively()
        val installer = EngineInstaller(
            filesDir = EngineFilesDir { context.filesDir },
            engineHandle = NativeEngineHandle { },
            httpFetcher = UrlHttpFetcher,
        )
        val installed = installer.install(name) { }
        assertTrue("Archive verification failed: ${installed.exceptionOrNull()}", installed.isSuccess)
        assertTrue(installer.verify(name) is InstallState.Installed)
        val engine = ParadeeEngine(context, SharedEspeakData(context))
        try {
            engine.ensureModelLoaded()
            for ((index, key) in ParadeeVoiceCatalog.VOICE_KEYS.withIndex()) {
                val audio = engine.synthesize(
                    text = "Hello! This is a Paradee voice, running entirely on Android.",
                    voiceId = "$name:$key",
                    speed = 1f,
                )
                assertEquals(24_000, audio.sampleRate)
                assertTrue("Short/empty audio for $key", audio.pcm.size > 8_000)
                val peak = audio.pcm.maxOf { abs(it.toInt()) }
                assertTrue("Silent audio for $key", peak > 100)
                if (index == 0) {
                    val buffer = ByteBuffer.allocate(audio.pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                    for (sample in audio.pcm) buffer.putShort(sample)
                    File(context.filesDir, "paradee-smoke.pcm").writeBytes(buffer.array())
                }
                println("PARADEE_OK voice=$key samples=${audio.pcm.size} peak=$peak")
            }
        } finally { engine.release() }
    }
}
