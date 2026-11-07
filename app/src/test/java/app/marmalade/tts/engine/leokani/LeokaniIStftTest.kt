package app.marmalade.tts.engine.leokani

import org.junit.Assert.assertEquals
import org.junit.Test

class LeokaniIStftTest {
    @Test fun silentSpectrogramIsSilentAtMultipleLengths() {
        for (frames in listOf(4, 8, 31)) {
            val zero = FloatArray(513 * frames)
            val result = LeokaniIStft.decode(zero, zero, frames)
            assertEquals(frames * 300, result.size)
            result.forEach { assertEquals(0.0f, it, 0.0f) }
        }
    }
}
