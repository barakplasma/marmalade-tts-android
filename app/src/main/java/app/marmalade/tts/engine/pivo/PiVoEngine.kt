package app.marmalade.tts.engine.pivo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import app.marmalade.tts.data.PiVoVoiceCatalog
import app.marmalade.tts.engine.EngineNotInstalledException
import app.marmalade.tts.engine.SynthAudio
import app.marmalade.tts.engine.TtsEngine
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * CPU-only PiVo 1.4M inference through four existing ONNX graphs.
 *
 * No Python, PyTorch or native DSP binaries. This is the inference procedure
 * published in PiVo's Apache-2.0 pivo/runtime.py, independently ported to Kotlin:
 * encoder -> duration/phase expansion -> causal decoder -> conditioning ->
 * stateful FARGAN subframe vocoder -> deemphasis. English frontend is CMUdict.
 */
@Singleton
class PiVoEngine @Inject constructor(
    @ApplicationContext private val context: Context,
) : TtsEngine {
    override val engineName = PiVoVoiceCatalog.ENGINE
    override val sampleRate = PiVoVoiceCatalog.SAMPLE_RATE
    override val maxInputChars = 150
    override val supportsNativeSpeed = false

    private val lock = Any()
    private val directory get() = File(context.filesDir, "engines/$engineName")
    private var sessions: Map<String, OrtSession> = emptyMap()
    private var frontend: PiVoFrontend? = null
    @Volatile private var loaded = false

    override fun isInstalled(): Boolean =
        listOf("encoder.onnx", "decoder.onnx", "conditioning.onnx",
            "vocoder_step.onnx", "metadata.json", "cmudict.dict").all {
            File(directory, it).length() > 0L
        }
    override fun isLoaded() = loaded

    override fun ensureModelLoaded() = synchronized(lock) {
        if (loaded) return@synchronized
        if (!isInstalled()) throw EngineNotInstalledException(engineName)
        val env = OrtEnvironment.getEnvironment()
        val created = HashMap<String, OrtSession>()
        try {
            for (name in listOf("encoder", "decoder", "conditioning", "vocoder_step")) {
                val options = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(2)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                }
                options.use { created[name] = env.createSession(File(directory, "$name.onnx").absolutePath, it) }
            }
            frontend = PiVoFrontend(File(directory, "cmudict.dict"))
            sessions = created
            loaded = true
        } catch (e: Throwable) {
            created.values.forEach { it.close() }
            throw e
        }
    }

    private val env get() = OrtEnvironment.getEnvironment()
    private fun tensor(values: FloatArray, vararg shape: Long): OnnxTensor =
        OnnxTensor.createTensor(
            env,
            ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder())
                .asFloatBuffer().apply { put(values); rewind() },
            shape,
        )
    private fun longs(values: LongArray, vararg shape: Long): OnnxTensor =
        OnnxTensor.createTensor(
            env,
            ByteBuffer.allocateDirect(values.size * 8).order(ByteOrder.nativeOrder())
                .asLongBuffer().apply { put(values); rewind() },
            shape,
        )

    /** Run one ONNX stage and copy its outputs before closing ORT tensors. */
    private fun run(name: String, inputs: Map<String, OnnxTensor>): Map<String, FloatArray> {
        try {
            sessions.getValue(name).run(inputs).use { results ->
                val output = HashMap<String, FloatArray>()
                for (key in sessions.getValue(name).outputInfo.keys) {
                    val tensor = results.get(key).get() as OnnxTensor
                    val count = tensor.info.shape.fold(1L) { a, b -> a * b }.toInt()
                    val values = FloatArray(count)
                    tensor.floatBuffer.apply { rewind(); get(values) }
                    output[key] = values
                }
                return output
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    private val states = listOf(
        "memory" to 256, "gru1" to 160, "gru2" to 128,
        "gru3" to 128, "frame_state" to 164,
    )

    private fun generate(ids: LongArray, voice: Int): ShortArray {
        if (ids.size <= 2) return ShortArray(0)
        val encoded = run("encoder", mapOf(
            "tokens" to longs(ids, 1, ids.size.toLong()),
            "voice" to longs(longArrayOf(voice.toLong()), 1),
            "lengths" to longs(longArrayOf(ids.size.toLong()), 1),
        ))
        val embeddings = encoded.getValue("encoded")
        val logs = encoded.getValue("log_durations")
        val durations = IntArray(ids.size) { i ->
            val raw = (exp(logs[i].toDouble().coerceIn(0.0, ln(201.0))) - 1.0)
                .roundToInt().coerceIn(0, 200)
            when {
                ids[i] <= 2L -> 0
                ids[i] >= 10L -> max(raw, 1)
                else -> raw
            }
        }
        val frameCount = durations.sum()
        if (frameCount < 1) return ShortArray(0)
        require(frameCount <= 3000) { "PiVo utterance exceeds 30-second budget" }
        val expanded = FloatArray(frameCount * 100)
        var frame = 0
        val normalizer = ln(201.0).toFloat()
        for (i in durations.indices) {
            val count = durations[i]
            for (step in 0 until count) {
                val base = frame * 100
                System.arraycopy(embeddings, i * 96, expanded, base, 96)
                val fraction = (step + 0.5f) / count
                expanded[base + 96] = 2 * fraction - 1
                expanded[base + 97] = sin(PI * fraction).toFloat()
                expanded[base + 98] = cos(PI * fraction).toFloat()
                expanded[base + 99] = ln(1f + count) / normalizer
                frame++
            }
        }

        // Full causal context in one graph call: acceptable for the short
        // clauses Marmalade sends; the model applies its 64-frame window.
        val zeroKey = FloatArray(0)
        val decoded = run("decoder", mapOf(
            "expanded" to tensor(expanded, 1, frameCount.toLong(), 100),
            "position" to longs(longArrayOf(0), *longArrayOf()),
            "key0" to tensor(zeroKey, 1, 2, 0, 24),
            "value0" to tensor(zeroKey, 1, 2, 0, 24),
            "key1" to tensor(zeroKey, 1, 2, 0, 24),
            "value1" to tensor(zeroKey, 1, 2, 0, 24),
        ))
        val features = decoded.getValue("features")
        val result = ShortArray(frameCount * 160)
        val stepState = states.associate { (name, width) -> name to FloatArray(width) }.toMutableMap()
        var deemphasis = 0f
        var offset = 0

        fun getFrame(i: Int): FloatArray =
            features.copyOfRange(i.coerceIn(0, frameCount - 1) * 20,
                i.coerceIn(0, frameCount - 1) * 20 + 20)
        fun period(frame: FloatArray): Long {
            val exponent = frame[18].coerceIn(-10f, 10f).toDouble() + 1.5
            return (256.0 / 2.0.pow(exponent)).roundToInt().coerceIn(32, 255).toLong()
        }
        for (i in 0 until frameCount) {
            val before = getFrame(max(0, i - 1))
            val current = getFrame(i)
            val after = getFrame((i + 1).coerceAtMost(frameCount - 1))
            val context = before + current + after
            val periods = longArrayOf(period(before), period(current), period(after))
            val conditioned = run("conditioning", mapOf(
                "features" to tensor(context, 1, 3, 20),
                "periods" to longs(periods, 1, 3),
            )).getValue("conditioning")
            for (sub in 0..3) {
                val input = HashMap<String, OnnxTensor>()
                input["conditioning"] = tensor(conditioned.copyOfRange(sub * 80, sub * 80 + 80), 1, 80)
                input["period"] = longs(longArrayOf(periods[1]), 1)
                for ((name, width) in states) {
                    input[name] = tensor(stepState.getValue(name), 1, width.toLong())
                }
                val generated = run("vocoder_step", input)
                for ((name, _) in states) stepState[name] = generated.getValue("${name}_out")
                for (sample in generated.getValue("excitation")) {
                    deemphasis = sample + 0.85f * deemphasis
                    result[offset++] = (deemphasis.coerceIn(-1f, 1f) * 32767f).roundToInt().toShort()
                }
            }
        }
        return result
    }

    override suspend fun synthesize(
        text: String,
        voiceId: String,
        speed: Float,
        phonemizationLanguage: String?,
        expressiveness: Float?,
        sentencePauseMs: Int,
    ): SynthAudio = withContext(Dispatchers.Default) {
        val key = voiceId.substringAfter(':')
        require(voiceId.startsWith("$engineName:") && key in PiVoVoiceCatalog.VOICE_KEYS)
        ensureModelLoaded()
        synchronized(lock) {
            val segments = text.trim().split(Regex("(?<=[.!?;])\\s+|(?<=,)\\s+"))
                .filter(String::isNotBlank)
                .flatMap { if (it.length <= maxInputChars) listOf(it) else it.chunked(maxInputChars) }
            val parts = segments.map { chunk ->
                generate(checkNotNull(frontend).encode(chunk), PiVoVoiceCatalog.VOICE_KEYS.indexOf(key))
            }
            val pcm = ShortArray(parts.sumOf { it.size })
            var offset = 0
            for (part in parts) {
                part.copyInto(pcm, offset)
                offset += part.size
            }
            SynthAudio(pcm, sampleRate)
        }
    }

    override fun release() = synchronized(lock) {
        loaded = false
        sessions.values.forEach { it.close() }
        sessions = emptyMap()
        frontend = null
    }
}
