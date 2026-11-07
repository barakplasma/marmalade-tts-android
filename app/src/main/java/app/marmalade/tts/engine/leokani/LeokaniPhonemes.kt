package app.marmalade.tts.engine.leokani

/**
 * Leokani's published eSpeak -> Kokoro pronunciation substitutions.
 * The model expects Kokoro-style phonemes, not raw IPA.
 */
internal object LeokaniPhonemes {
    private val replacements = listOf(
        "ʔˌn̩" to "tn", "ʔn̩" to "tn", "ʔn" to "tn",
        "a^ɪ" to "I", "a^ʊ" to "W", "d^ʒ" to "ʤ",
        "e^ɪ" to "A", "t^ʃ" to "ʧ", "ɔ^ɪ" to "Y",
        "ə^l" to "ᵊl", "ʲo" to "jo", "ʲə" to "jə",
        "ɚ" to "əɹ", "ʔ" to "t", "e" to "A", "ʲ" to "",
        "r" to "ɹ", "x" to "k", "ç" to "k", "ɐ" to "ə", "ɬ" to "l", "̃" to "",
    )

    fun fromEspeak(ipa: String): String {
        var result = ipa.trim()
        for ((old, new) in replacements) result = result.replace(old, new)
        result = Regex("(\\S)̩").replace(result) { "ᵊ" + it.groupValues[1] }.replace("̩", "")
        for ((old, new) in listOf(
            "o^ʊ" to "O", "ɜːɹ" to "ɜɹ", "ɜː" to "ɜɹ",
            "ɪə" to "iə", "ː" to "", "o" to "ɔ", "^" to "",
        )) result = result.replace(old, new)
        return result
    }

    /** Unicode combining marks belong to their preceding phoneme. */
    fun toIds(phonemes: String, vocabulary: Map<String, Long>): LongArray {
        val ids = ArrayList<Long>()
        var offset = 0
        while (offset < phonemes.length) {
            val point = phonemes.codePointAt(offset)
            val cluster = StringBuilder().appendCodePoint(point)
            offset += Character.charCount(point)
            while (offset < phonemes.length) {
                val next = phonemes.codePointAt(offset)
                val cat = Character.getType(next)
                if (cat != Character.NON_SPACING_MARK.toInt() &&
                    cat != Character.COMBINING_SPACING_MARK.toInt() &&
                    cat != Character.ENCLOSING_MARK.toInt()
                ) break
                cluster.appendCodePoint(next)
                offset += Character.charCount(next)
            }
            vocabulary[cluster.toString()]?.let { ids.add(it) }
        }
        return ids.toLongArray()
    }
}
