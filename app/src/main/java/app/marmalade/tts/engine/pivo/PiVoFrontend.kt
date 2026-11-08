package app.marmalade.tts.engine.pivo

import java.io.File
import java.text.Normalizer
import java.util.Locale

/**
 * English CMUdict-to-ARPAbet frontend for PiVo's 128-entry phoneme vocabulary.
 * Avoids guessing unknown English pronunciations: falls back to spelling
 * their letters using the provided dictionary, like upstream's optional mode.
 */
internal class PiVoFrontend(dictionaryFile: File) {
    private val special = listOf("<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":")
    private val consonants = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split(" ")
    private val vowels = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split(" ")
    private val symbols = special + consonants + vowels.flatMap { v -> listOf("${v}0", "${v}1", "${v}2") }
    private val symbolIds = symbols.withIndex().associate { it.value to it.index.toLong() }
    private val lexicon = HashMap<String, List<String>>(110_000)
    private val wordRegex = Regex("[A-Za-z]+(?:'[A-Za-z]+)*|[.,?!;:]")
    private val digitNames = listOf("zero","one","two","three","four","five","six","seven","eight","nine")

    init {
        dictionaryFile.forEachLine { line ->
            val cols = line.substringBefore('#').trim().split(Regex("\\s+"))
            if (cols.size < 2) return@forEachLine
            val key = cols[0].replace(Regex("\\(\\d+\\)$"), "").lowercase(Locale.US)
            if (key !in lexicon) lexicon[key] = cols.drop(1)
        }
        check(lexicon.size > 50_000) { "PiVo CMU dictionary missing or corrupt" }
    }

    fun encode(text: String): LongArray {
        var clean = Normalizer.normalize(text, Normalizer.Form.NFKC)
            .replace('’', '\'').replace('‘', '\'').replace('“', '"').replace('”', '"')
            .replace("—", ",").replace("–", ",")
        clean = Regex("\\b\\d+\\b").replace(clean) { match ->
            match.value.map { digitNames[it.digitToInt()] }.joinToString(" ")
        }
        clean = Regex("\\b[A-Z]{2,5}\\b").replace(clean) { match ->
            if (match.value == "NASA") match.value.lowercase(Locale.US)
            else match.value.toCharArray().joinToString(" ")
        }
        val tokens = ArrayList<Long>()
        tokens += 1L // BOS
        for (match in wordRegex.findAll(clean)) {
            val item = match.value
            val direct = symbolIds[item]
            if (direct != null) { tokens += direct; continue }
            val word = item.lowercase(Locale.US)
            val pron = lexicon[word] ?: word.flatMap { c ->
                lexicon[c.toString()] ?: emptyList()
            }
            if (pron.isEmpty()) continue
            if (tokens.last() != 1L && tokens.last() != 3L) tokens += 3L // SPACE
            for (p in pron) symbolIds[p]?.let(tokens::add)
        }
        tokens += 2L // EOS
        check(tokens.size <= 512) { "PiVo input exceeds 512 phoneme tokens" }
        return tokens.toLongArray()
    }
}
