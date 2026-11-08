package app.marmalade.tts.engine.paradee

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import app.marmalade.tts.data.ParadeeVoiceCatalog
import app.marmalade.tts.engine.EngineNotInstalledException
import app.marmalade.tts.engine.SynthAudio
import app.marmalade.tts.engine.TtsEngine
import app.marmalade.tts.phonemizer.EnPhonemeFixups
import app.marmalade.tts.phonemizer.EspeakPhonemizer
import app.marmalade.tts.phonemizer.SharedEspeakData
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import kotlin.math.roundToInt

/** Full Paradee 1.2 inference in one ONNX session, with no Python on Android. */
@Singleton
class ParadeeEngine @Inject constructor(
    @ApplicationContext private val context: Context,
    private val sharedEspeak: SharedEspeakData,
) : TtsEngine {
    override val engineName = ParadeeVoiceCatalog.ENGINE
    override val sampleRate = ParadeeVoiceCatalog.SAMPLE_RATE
    override val maxInputChars = 180
    override val supportsNativeSpeed = false

    private val lock = Any()
    private val directory get() = File(context.filesDir, "engines/$engineName")
    private var session: OrtSession? = null
    private var phonemizer: EspeakPhonemizer? = null
    private var vocabulary: Map<String, Long> = emptyMap()
    @Volatile private var loaded = false

    override fun isInstalled(): Boolean = File(directory, "model.onnx").length() > 0L &&
        File(directory, "config.json").length() > 0L
    override fun isLoaded() = loaded

    override fun ensureModelLoaded() = synchronized(lock) {
        if (loaded) return@synchronized
        if (!isInstalled()) throw EngineNotInstalledException(engineName)
        val config = JSONObject(File(directory, "config.json").readText())
        val vocab = HashMap<String, Long>()
        val entries = config.getJSONObject("vocab")
        val keys = entries.keys()
        while (keys.hasNext()) { val key = keys.next(); vocab[key] = entries.getLong(key) }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        var created: OrtSession? = null
        var espeak: EspeakPhonemizer? = null
        try {
            options.use {
                val model = File(directory, "model.onnx")
                created = if (Build.SUPPORTED_ABIS.firstOrNull()?.startsWith("armeabi") == true) {
                    OrtEnvironment.getEnvironment().createSession(model.readBytes(), it)
                } else {
                    OrtEnvironment.getEnvironment().createSession(model.absolutePath, it)
                }
            }
            espeak = EspeakPhonemizer(
                libPath = EspeakPhonemizer.APK_LIB_NAME,
                dataPath = sharedEspeak.ensure().absolutePath,
                fixupModel = EnPhonemeFixups.Model.KOKORO,
            )
            check(espeak.open() >= 0) { "Could not open eSpeak for Paradee" }
            session = created
            phonemizer = espeak
            vocabulary = vocab
            loaded = true
        } catch (t: Throwable) {
            created?.close()
            espeak?.close()
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
        val key = voiceId.substringAfter(':', missingDelimiterValue = "")
        val voiceIndex = ParadeeVoiceCatalog.VOICE_KEYS.indexOf(key)
        require(voiceId.startsWith("$engineName:") && voiceIndex >= 0) {
            "Unknown Paradee voice: $voiceId"
        }
        ensureModelLoaded()
        synchronized(lock) {
            val onnx = checkNotNull(session)
            val espeak = checkNotNull(phonemizer)
            val british = key.startsWith("b")
            val lang = if (british) "en-gb" else "en-us"
            val ort = OrtEnvironment.getEnvironment()
            val chunks = text.trim()
                .split(Regex("(?<=[.!?;])\\s+|(?<=,)\\s+"))
                .filter(String::isNotBlank)
                .flatMap { if (it.length <= maxInputChars) listOf(it) else it.chunked(maxInputChars) }
            val parts = ArrayList<ShortArray>()
            for (chunk in chunks) {
                val ipa = espeak.phonemize(chunk, lang, tie = false)
                val phonemes = ParadeePhonemes.fromEspeak(ipa, british)
                val tokens = ParadeePhonemes.toIds(phonemes, vocabulary)
                if (tokens.isEmpty()) continue
                for (frame in tokens.toList().chunked(500)) {
                    val ids = longArrayOf(0L, *frame.toLongArray(), 0L)
                    val idBuf = ByteBuffer.allocateDirect(ids.size * 8)
                        .order(ByteOrder.nativeOrder()).asLongBuffer().apply { put(ids); rewind() }
                    val speedBuf = ByteBuffer.allocateDirect(4)
                        .order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(1.0f); rewind() }
                    val voiceBuf = ByteBuffer.allocateDirect(8)
                        .order(ByteOrder.nativeOrder()).asLongBuffer().apply { put(voiceIndex.toLong()); rewind() }
                    OnnxTensor.createTensor(ort, idBuf, longArrayOf(1L, ids.size.toLong())).use { input ->
                        OnnxTensor.createTensor(ort, speedBuf, longArrayOf(1L)).use { speedTensor ->
                            OnnxTensor.createTensor(ort, voiceBuf, longArrayOf(1L)).use { voiceTensor ->
                                onnx.run(mapOf(
                                    "input_ids" to input,
                                    "speed" to speedTensor,
                                    "voice" to voiceTensor,
                                )).use { result ->
                                    val wave = result.get("waveform").get() as OnnxTensor
                                    val shape = wave.info.shape
                                    check(shape.size == 2 && shape[0] == 1L && shape[1] > 0L) {
                                        "Unexpected Paradee waveform: ${shape.contentToString()}"
                                    }
                                    val samples = FloatArray(shape[1].toInt())
                                    wave.floatBuffer.apply { rewind(); get(samples) }
                                    parts += ShortArray(samples.size) { i ->
                                        (samples[i].coerceIn(-1f, 1f) * 32767f).roundToInt().toShort()
                                    }
                                }
                            }
                        }
                    }
                }
            }
            val pcm = ShortArray(parts.sumOf { it.size })
            var at = 0
            for (part in parts) { part.copyInto(pcm, at); at += part.size }
            SynthAudio(pcm, sampleRate)
        }
    }

    override fun release() = synchronized(lock) {
        loaded = false
        session?.close()
        session = null
        phonemizer?.close()
        phonemizer = null
        vocabulary = emptyMap()
    }
}
