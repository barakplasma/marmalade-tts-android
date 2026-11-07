package app.marmalade.tts.engine.leokani

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Converts Leokani's 1024-point "same"-padded STFT magnitude/phase
 * to PCM float samples. The input tensors are flattened [513, frames].
 * No Python, JNI or additional Android dependencies.
 */
internal object LeokaniIStft {
    private const val FFT = 1024
    private const val BINS = FFT / 2 + 1
    private const val HOP = 300
    private const val PAD = (FFT - HOP) / 2
    private val hann = DoubleArray(FFT) { 0.5 - 0.5 * cos(2.0 * PI * it / FFT) }
    private val hann2 = DoubleArray(FFT) { hann[it] * hann[it] }

    fun decode(mag: FloatArray, phase: FloatArray, frames: Int): FloatArray {
        require(frames > 0 && mag.size == BINS * frames && phase.size == mag.size) {
            "Leokani spectrogram mismatch"
        }
        val length = (frames - 1) * HOP + FFT
        val sum = DoubleArray(length)
        val env = DoubleArray(length)
        val real = DoubleArray(FFT)
        val imag = DoubleArray(FFT)
        for (t in 0 until frames) {
            for (k in 0 until BINS) {
                val idx = k * frames + t
                val angle = phase[idx].toDouble()
                real[k] = mag[idx].toDouble() * cos(angle)
                imag[k] = mag[idx].toDouble() * sin(angle)
            }
            for (k in 1 until FFT / 2) {
                real[FFT - k] = real[k]
                imag[FFT - k] = -imag[k]
            }
            imag[0] = 0.0
            imag[FFT / 2] = 0.0
            fftInPlace(real, imag)
            val offset = t * HOP
            for (j in 0 until FFT) {
                sum[offset + j] += real[j] * hann[j]
                env[offset + j] += hann2[j]
            }
        }
        return FloatArray(frames * HOP) { i ->
            val k = i + PAD
            (sum[k] / max(env[k], 1e-11)).toFloat()
        }
    }

    private fun fftInPlace(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while ((j and bit) != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                var tmp = re[i]
                re[i] = re[j]
                re[j] = tmp
                tmp = im[i]
                im[i] = im[j]
                im[j] = tmp
            }
        }
        var size = 2
        while (size <= n) {
            val angle = 2.0 * PI / size
            val stepRe = cos(angle)
            val stepIm = sin(angle)
            val half = size / 2
            var start = 0
            while (start < n) {
                var wr = 1.0
                var wi = 0.0
                for (k in 0 until half) {
                    val even = start + k
                    val odd = even + half
                    val tr = wr * re[odd] - wi * im[odd]
                    val ti = wr * im[odd] + wi * re[odd]
                    val er = re[even]
                    val ei = im[even]
                    re[even] = er + tr
                    im[even] = ei + ti
                    re[odd] = er - tr
                    im[odd] = ei - ti
                    val nextWr = wr * stepRe - wi * stepIm
                    wi = wr * stepIm + wi * stepRe
                    wr = nextWr
                }
                start += size
            }
            size = size shl 1
        }
        for (i in 0 until n) {
            re[i] /= n.toDouble()
            im[i] /= n.toDouble()
        }
    }
}
