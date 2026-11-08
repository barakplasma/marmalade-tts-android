package app.marmalade.tts.engine.phonon

import android.content.Context
import android.os.Build
import android.system.Os
import app.marmalade.tts.data.PhononVoiceCatalog
import app.marmalade.tts.engine.EngineNotInstalledException
import app.marmalade.tts.engine.SynthAudio
import app.marmalade.tts.engine.TtsEngine
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

/**
 * Phonon 31m TTS via Gradium's native Rust CPU backend, exposed through
 * libptts_ffi.so and the upstream Apache-2.0 Kotlin JNA wrapper.
 *
 * The checkpoint is downloaded separately; the lib is packaged only in
 * ARM64/x86_64 builds. Neither Python nor a local HTTP service is used.
 */
@Singleton
class PhononEngine @Inject constructor(
    @ApplicationContext private val context: Context,
) : TtsEngine {
    override val engineName = PhononVoiceCatalog.ENGINE
    override val sampleRate = PhononVoiceCatalog.SAMPLE_RATE
    override val maxInputChars = 500
    override val supportsNativeSpeed = false

    private val lock = Any()
    private val modelDir get() = File(context.filesDir, "engines/$engineName")
    private var model: PhononTTS? = null
    @Volatile private var loaded = false

    private fun nativeSupported() = Build.SUPPORTED_ABIS.any { it == "arm64-v8a" || it == "x86_64" }

    override fun isInstalled(): Boolean = nativeSupported() &&
        listOf("config.json", "tokenizer.json", "model.q8.gguf", "default-voice.safetensors").all {
            File(modelDir, it).length() > 0L
        }

    override fun isLoaded() = loaded

    override fun ensureModelLoaded() = synchronized(lock) {
        if (loaded) return@synchronized
        if (!isInstalled()) throw EngineNotInstalledException(engineName)
        // Use a bounded CPU pool rather than all big + LITTLE cores.
        Os.setenv("RAYON_NUM_THREADS", "2", true)
        var created: PhononTTS? = null
        try {
            created = PhononTTS(modelDir.absolutePath, "en")
            created.setVoice("default")
            model = created
            loaded = true
        } catch (t: Throwable) {
            created?.close()
            throw t
        }
    }

    override suspend fun synthesize(
        text: String,
        voiceId: String,
        speed: Float,
        phonemizationLanguage: String?,
        expressiveness: Float?,
        sentencePauseMs: Int,
    ): SynthAudio = withContext(Dispatchers.Default) {
        require(voiceId == PhononVoiceCatalog.DEFAULT_VOICE_ID) { "Unknown Phonon voice: $voiceId" }
        ensureModelLoaded()
        synchronized(lock) {
            val tts = checkNotNull(model)
            val chunks = ArrayList<ShortArray>()
            // Gradium's FFI streams float PCM fragments as soon as generated.
            tts.speak(text) { frames ->
                chunks += ShortArray(frames.size) { idx ->
                    (frames[idx].coerceIn(-1f, 1f) * 32767f).roundToInt().toShort()
                }
                true
            }
            val combined = ShortArray(chunks.sumOf { it.size })
            var at = 0
            for (chunk in chunks) { chunk.copyInto(combined, at); at += chunk.size }
            SynthAudio(combined, sampleRate)
        }
    }

    override fun release() = synchronized(lock) {
        loaded = false
        model?.close()
        model = null
    }
}
