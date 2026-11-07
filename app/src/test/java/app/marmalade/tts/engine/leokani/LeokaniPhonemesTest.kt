package app.marmalade.tts.engine.leokani

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class LeokaniPhonemesTest {
    @Test fun convertsEspeakDiphthongsAndAffricates() {
        assertEquals("I W ʧ əɹ", LeokaniPhonemes.fromEspeak("a^ɪ a^ʊ t^ʃ ɚ"))
    }

    @Test fun groupsCombiningMarksAsOneVocabularySymbol() {
        val vocabulary = mapOf("a̩" to 5L, "b" to 6L)
        assertArrayEquals(longArrayOf(5L, 6L), LeokaniPhonemes.toIds("a̩b", vocabulary))
    }

    @Test fun filtersCharactersMissingFromVocabulary() {
        assertArrayEquals(longArrayOf(3L), LeokaniPhonemes.toIds("a?z", mapOf("a" to 3L)))
    }
}
