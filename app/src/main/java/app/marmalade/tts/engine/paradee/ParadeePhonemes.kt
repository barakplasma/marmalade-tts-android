package app.marmalade.tts.engine.paradee

/**
 * eSpeak IPA -> Misaki/Kokoro phonemes for Paradee 1.2.
 * Based on upstream Paradee's Apache-2.0 web/misaki.js and Misaki's
 * EspeakFallback conventions. Keep American and British rules distinct.
 */
internal object ParadeePhonemes {
    private val replacements = listOf(
        "ʔˌn̩" to "ʔn", "ʔn̩" to "ʔn",
        "aɪ" to "I", "aʊ" to "W", "dʒ" to "ʤ",
        "eɪ" to "A", "e" to "A", "tʃ" to "ʧ", "ɔɪ" to "Y",
        "ʲo" to "jo", "ʲə" to "jə", "ʲ" to "",
        "ɚ" to "əɹ", "r" to "ɹ",
        "x" to "k", "ç" to "k", "ɬ" to "l", "̃" to "",
    )

    private val wordBoundary = """(?=[\s;:,.!?—…"“”()]|$)""".toRegex()

    fun fromEspeak(ipa: String, british: Boolean): String {
        var text = ipa.trim()
        if (british) text = text.replace("eə", "ɛː")
        for ((from, to) in replacements) text = text.replace(from, to)
        text = Regex("(\\S)̩").replace(text) { "ᵊ" + it.groupValues[1] }.replace("̩", "")
        text = Regex("əl(?=[\\s;:,.!?—…\"“”()]|$)").replace(text, "ᵊl")
        // Keep the isolated weak article "a" (and British "an") as ɐ.
        text = if (british) {
            Regex("ɐ(?!n?(?:[\\s;:,.!?—…\"“”()]|$))").replace(text, "ə")
        } else {
            Regex("ɐ(?![\\s;:,.!?—…\"“”()]|$)").replace(text, "ə")
        }
        if (british) {
            text = text.replace("iə", "ɪə").replace("əʊ", "Q")
            text = Regex("(^|\\s)ðɪ(?=\\s|$)").replace(text, "$1ði")
        } else {
            text = text.replace("oʊ", "O")
                .replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ")
                .replace("ɪə", "iə").replace("ː", "")
        }
        return text.replace("o", "ɔ").replace("ɾ", "T").replace("ʔ", "t")
    }

    /** Preserve Unicode combining marks with the phoneme that precedes them. */
    fun toIds(text: String, vocab: Map<String, Long>): LongArray {
        val ids = ArrayList<Long>()
        var index = 0
        while (index < text.length) {
            val codepoint = text.codePointAt(index)
            val cluster = StringBuilder().appendCodePoint(codepoint)
            index += Character.charCount(codepoint)
            while (index < text.length) {
                val next = text.codePointAt(index)
                val type = Character.getType(next)
                if (type != Character.NON_SPACING_MARK.toInt() &&
                    type != Character.COMBINING_SPACING_MARK.toInt() &&
                    type != Character.ENCLOSING_MARK.toInt()) break
                cluster.appendCodePoint(next)
                index += Character.charCount(next)
            }
            vocab[cluster.toString()]?.let(ids::add)
        }
        return ids.toLongArray()
    }
}
