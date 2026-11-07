package app.marmalade.tts.engine.leokani

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import app.marmalade.tts.data.LeokaniVoiceCatalog
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

/**
 * Optional offline English TTS via Leokani 7M int8 ONNX.
 * One ORT graph produces STFT magnitude/phase; pure Kotlin iSTFT yields PCM.
 * Install layout: filesDir/engines/leokani-tts-7m/model.onnx + config.json.
 */
@Singleton
class LeokaniEngine @Inject constructor(
    @ApplicationContext private val context: Context,
    private val sharedEspeak: SharedEspeakData,
) : TtsEngine {
    override val engineName: String = LeokaniVoiceCatalog.ENGINE
    override val sampleRate: Int = LeokaniVoiceCatalog.SAMPLE_RATE
    override val maxInputChars: Int = 160
    override val supportsNativeSpeed: Boolean = false

    private val lock = Any()
    private val directory: File get() = File(context.filesDir, "engines/" + engineName)
    private var session: OrtSession? = null
    private var phonemizer: EspeakPhonemizer? = null
    private var vocabulary: Map<String, Long> = emptyMap()
    @Volatile private var loaded = false

    override fun isInstalled(): Boolean =
        File(directory, "model.onnx").length() > 0L &&
            File(directory, "config.json").length() > 0L

    override fun isLoaded(): Boolean = loaded

    override fun ensureModelLoaded() = synchronized(lock) {
        if (loaded) return@synchronized
        if (!isInstalled()) throw EngineNotInstalledException(engineName)
        val config = JSONObject(File(directory, "config.json").readText())
        val entries = config.getJSONObject("vocab")
        val vocab = HashMap<String, Long>()
        val keys = entries.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            vocab[k] = entries.getLong(k)
        }
        val ort = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        var created: OrtSession? = null
        var espeak: EspeakPhonemizer? = null
        try {
            opts.use { options ->
                val model = File(directory, "model.onnx")
                created = if (Build.SUPPORTED_ABIS.firstOrNull()?.startsWith("armeabi") == true) {
                    ort.createSession(model.readBytes(), options)
                } else {
                    ort.createSession(model.absolutePath, options)
                }
            }
            espeak = EspeakPhonemizer(
                libPath = EspeakPhonemizer.APK_LIB_NAME,
                dataPath = sharedEspeak.ensure().absolutePath,
                fixupModel = EnPhonemeFixups.Model.KOKORO,
            )
            check(espeak.open() >= 0) { "Could not open eSpeak for Leokani" }
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
        require(voiceId == LeokaniVoiceCatalog.DEFAULT_VOICE_ID) {
            "Unknown Leokani voice: " + voiceId
        }
        ensureModelLoaded()
        synchronized(lock) {
            val onnx = checkNotNull(session)
            val espeak = checkNotNull(phonemizer)
            val ort = OrtEnvironment.getEnvironment()
            val pieces = text.trim()
                .split(Regex("(?<=[.!?;])\\s+|(?<=,)\\s+"))
                .filter { it.isNotBlank() }
                .flatMap { part ->
                    if (part.length <= maxInputChars) listOf(part) else part.chunked(maxInputChars)
                }
            val segments = ArrayList<ShortArray>()
            for (piece in pieces) {
                val ipa = espeak.phonemize(piece, "en-us", tie = true)
                val normalized = LeokaniPhonemes.fromEspeak(ipa)
                val tokenIds = LeokaniPhonemes.toIds(normalized, vocabulary)
                if (tokenIds.isEmpty()) continue
                for (part in tokenIds.toList().chunked(400)) {
                    val input = longArrayOf(0L, *part.toLongArray(), 0L)
                    val idsBuf = ByteBuffer.allocateDirect(input.size * 8)
                        .order(ByteOrder.nativeOrder()).asLongBuffer()
                        .apply { put(input); rewind() }
                    val speedBuf = ByteBuffer.allocateDirect(4)
                        .order(ByteOrder.nativeOrder()).asFloatBuffer()
                        .apply { put(1.0f); rewind() }

                    OnnxTensor.createTensor(ort, idsBuf, longArrayOf(1L, input.size.toLong())).use { ids ->
                        OnnxTensor.createTensor(ort, speedBuf, longArrayOf(1L)).use { spd ->
                            onnx.run(mapOf("ids" to ids, "speed" to spd)).use { results ->
                                val magnitude = results.get("mag").get() as OnnxTensor
                                val angles = results.get("phase").get() as OnnxTensor
                                val shape = magnitude.info.shape
                                check(shape.size == 3 && shape[0] == 1L && shape[1] == 513L) {
                                    "Unexpected Leokani output dimensions: " + shape.contentToString()
                                }
                                val frames = shape[2].toInt()
                                val mag = FloatArray(513 * frames)
                                val phase = FloatArray(513 * frames)
                                magnitude.floatBuffer.apply { rewind(); get(mag) }
                                angles.floatBuffer.apply { rewind(); get(phase) }
                                val samples = LeokaniIStft.decode(mag, phase, frames)
                                segments += ShortArray(samples.size) { i ->
                                    (samples[i].coerceIn(-1f, 1f) * 32767f).roundToInt().toShort()
                                }
                            }
                        }
                    }
                }
            }
            val result = ShortArray(segments.sumOf { it.size })
            var offset = 0
            for (segment in segments) {
                segment.copyInto(result, offset)
                offset += segment.size
            }
            SynthAudio(result, sampleRate)
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
