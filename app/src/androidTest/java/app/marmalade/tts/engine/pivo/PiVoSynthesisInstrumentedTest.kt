package app.marmalade.tts.engine.pivo

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import app.marmalade.tts.data.PiVoVoiceCatalog
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

@RunWith(AndroidJUnit4::class)
class PiVoSynthesisInstrumentedTest {
    @Test fun installsFourGraphsAndSynthesizesBothVoices() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = PiVoVoiceCatalog.ENGINE
        File(context.filesDir, "engines/$name").deleteRecursively()
        val installer = EngineInstaller(
            filesDir = EngineFilesDir { context.filesDir },
            engineHandle = NativeEngineHandle { },
            httpFetcher = UrlHttpFetcher,
        )
        val result = installer.install(name) { }
        assertTrue("PiVo download failed: ${result.exceptionOrNull()}", result.isSuccess)
        assertTrue("PiVo layout failed", installer.verify(name) is InstallState.Installed)
        val engine = PiVoEngine(context)
        try {
            engine.ensureModelLoaded()
            for ((i, key) in PiVoVoiceCatalog.VOICE_KEYS.withIndex()) {
                val audio = engine.synthesize(
                    "Hello! This is PiVo speaking clearly from an Android phone.",
                    "$name:$key",
                    1f,
                )
                assertEquals(16_000, audio.sampleRate)
                assertTrue("Short audio for $key", audio.pcm.size > 8_000)
                val peak = audio.pcm.maxOf { abs(it.toInt()) }
                assertTrue("Silent PCM for $key", peak > 100)
                if (i == 0) {
                    val pcm = ByteBuffer.allocate(audio.pcm.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                    audio.pcm.forEach { pcm.putShort(it) }
                    File(context.filesDir, "pivo-smoke.pcm").writeBytes(pcm.array())
                }
                println("PIVO_OK voice=$key samples=${audio.pcm.size} peak=$peak")
            }
        } finally { engine.release() }
    }
}
