package app.marmalade.tts.engine.paradee

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ParadeePhonemesTest {
    @Test fun americanDiphthongsAndAffricates() {
        assertEquals("I W ʤ ʧ Y O", ParadeePhonemes.fromEspeak("aɪ aʊ dʒ tʃ ɔɪ oʊ", british = false))
    }
    @Test fun britishVowelsKeepTheirLength() {
        assertEquals("ɛː Q", ParadeePhonemes.fromEspeak("eə əʊ", british = true))
    }
    @Test fun mapsTokensAndPunctuationFromConfig() {
        val ids = ParadeePhonemes.toIds("hi!", mapOf("h" to 50L, "i" to 51L, "!" to 5L))
        assertTrue(ids.contentEquals(longArrayOf(50L, 51L, 5L)))
    }
}
