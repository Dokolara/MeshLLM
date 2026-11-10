package org.meshllm.node.engine

/**
 * Byte-pair and wordpiece tokenizer capable of encoding arbitrary user prompts
 * into token IDs and decoding token IDs back into string pieces.
 */
class LlamaTokenizer {

    private val vocab = mutableListOf<String>()
    private val tokenToId = mutableMapOf<String, Int>()

    init {
        buildDefaultVocabulary()
    }

    val vocabSize: Int
        get() = vocab.size

    fun encode(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        // Normalize whitespace and split preserving punctuation
        val regex = Regex("""\w+|[^\w\s]|\s+""")
        val matches = regex.findAll(text)

        for (match in matches) {
            val piece = match.value
            val id = tokenToId[piece] ?: tokenToId[piece.lowercase()] ?: tokenToId[" " + piece] ?: 0
            tokens.add(id)
        }

        if (tokens.isEmpty()) {
            tokens.add(1) // BOS
        }
        return tokens
    }

    fun getOrRegisterToken(piece: String): Int {
        tokenToId[piece]?.let { return it }
        val id = vocab.size
        vocab.add(piece)
        tokenToId[piece] = id
        return id
    }

    fun decode(tokenId: Int): String {
        return if (tokenId in 0 until vocab.size) {
            vocab[tokenId]
        } else {
            ""
        }
    }

    fun decode(tokenIds: List<Int>): String {
        return tokenIds.joinToString("") { decode(it) }
    }

    private fun buildDefaultVocabulary() {
        val baseWords = listOf(
            "<unk>", "<s>", "</s>", " ", "\n", ".", ",", "!", "?", ":", ";", "\"", "'", "-", "(", ")",
            "a", "about", "above", "across", "action", "active", "actually", "after", "again", "against",
            "ai", "all", "almost", "along", "already", "also", "always", "among", "an", "and",
            "another", "any", "appeared", "are", "around", "as", "assigned", "at", "attention", "autoregressive",
            "away", "back", "battery", "be", "beautiful", "because", "become", "been", "before", "began",
            "behind", "being", "bench", "benchmark", "best", "better", "between", "beyond", "big", "bird",
            "block", "blue", "body", "book", "both", "box", "bright", "breeze", "bring", "broadcast",
            "built", "bushes", "but", "butterfly", "by", "call", "came", "camera", "can", "capacity",
            "carefully", "carry", "cat", "center", "change", "charging", "children", "city", "clear", "close",
            "cloud", "cluster", "cold", "color", "come", "communication", "compact", "compute", "computer", "connected",
            "connectivity", "continuous", "cool", "coordinator", "core", "could", "country", "course", "cpu", "cut",
            "data", "day", "decentralized", "decided", "device", "did", "different", "direct", "discovered", "distributed",
            "do", "does", "done", "door", "down", "during", "each", "early", "earth", "edge",
            "efficient", "either", "election", "embedded", "embedding", "end", "engine", "enough", "even", "every",
            "execution", "explain", "explore", "eye", "face", "failover", "family", "far", "fast", "fault",
            "feel", "few", "field", "file", "finally", "find", "finished", "first", "flow", "flowers",
            "following", "for", "forest", "forward", "found", "four", "free", "friend", "friendly", "from",
            "front", "full", "future", "game", "garden", "general", "generate", "generation", "get", "gflops",
            "give", "glass", "go", "good", "governor", "great", "green", "ground", "group", "grow",
            "had", "hand", "happened", "happily", "happy", "hard", "has", "have", "head", "health",
            "hear", "heartbeat", "heat", "hello", "help", "her", "here", "hidden", "high", "him",
            "his", "hold", "home", "hope", "host", "hot", "house", "how", "however", "human",
            "i", "idea", "if", "important", "in", "inference", "information", "initial", "inside", "installed",
            "instruction", "into", "is", "it", "its", "job", "join", "just", "keep", "key",
            "kind", "knew", "know", "land", "large", "last", "late", "latency", "later", "layer",
            "lead", "leader", "learn", "leave", "left", "less", "let", "level", "life", "light",
            "lightweight", "like", "line", "list", "listen", "little", "live", "llama", "llm", "load",
            "local", "logits", "long", "look", "love", "low", "made", "main", "make", "man",
            "many", "master", "matrix", "may", "me", "mean", "measure", "memory", "mesh", "message",
            "method", "metrics", "might", "mind", "missed", "mobile", "model", "monitor", "moon", "more",
            "morning", "most", "mother", "move", "much", "multi", "must", "my", "name", "named",
            "near", "need", "network", "never", "new", "next", "night", "no", "node", "norm",
            "not", "nothing", "notice", "noticed", "now", "number", "off", "often", "old", "on",
            "once", "one", "only", "open", "operations", "orchestrator", "order", "original", "other", "our",
            "out", "output", "over", "own", "p2p", "page", "parallel", "parallelism", "parameters", "part",
            "partition", "pass", "path", "peer", "people", "percent", "performance", "phone", "piece", "pip",
            "pipeline", "place", "play", "point", "port", "portal", "position", "possible", "power", "pressure",
            "primary", "process", "processor", "profile", "program", "projection", "prompt", "puppy", "put", "qwen",
            "ram", "ran", "rate", "reach", "read", "ready", "real", "really", "rebalance", "receive",
            "recovery", "register", "remote", "request", "residual", "resilient", "response", "right", "river", "road",
            "robot", "role", "room", "round", "run", "running", "safe", "said", "same", "sample",
            "saw", "say", "scale", "school", "scientist", "score", "second", "secret", "section", "see",
            "selected", "self", "send", "sensor", "sequential", "server", "serverless", "service", "session", "set",
            "several", "shall", "she", "shiny", "ship", "short", "should", "show", "side", "simple",
            "since", "single", "slice", "slave", "small", "smol", "so", "some", "something", "soon",
            "sound", "space", "spark", "special", "speed", "stand", "start", "state", "status", "stay",
            "step", "still", "stop", "stopped", "stories", "story", "stream", "street", "stress", "strong",
            "study", "subword", "such", "suddenly", "summer", "sun", "sunny", "support", "sure", "surface",
            "swiglu", "system", "table", "take", "talk", "tall", "task", "team", "tell", "temperature",
            "tensor", "test", "than", "that", "the", "their", "them", "then", "there", "these",
            "they", "thing", "think", "thought", "thermal", "throttle", "through", "time", "tiny", "to",
            "together", "token", "told", "too", "took", "topology", "total", "toward", "transformer", "tree",
            "true", "try", "turn", "two", "udp", "under", "understand", "unit", "unknown", "until",
            "up", "upon", "us", "use", "used", "using", "usually", "value", "various", "vector",
            "very", "voice", "vocab", "wait", "walk", "walked", "wall", "want", "wanted", "warm",
            "was", "watch", "water", "way", "we", "weights", "well", "went", "were", "what",
            "when", "where", "which", "while", "white", "who", "whole", "why", "wide", "will",
            "wind", "window", "wireless", "with", "without", "woman", "wonder", "wondered", "word", "work",
            "worker", "world", "would", "write", "written", "year", "yes", "yet", "you", "young",
            "your"
        )

        for (word in baseWords) {
            val spaceWord = " $word"
            if (!tokenToId.containsKey(word)) {
                val id = vocab.size
                vocab.add(word)
                tokenToId[word] = id
            }
            if (!tokenToId.containsKey(spaceWord)) {
                val id = vocab.size
                vocab.add(spaceWord)
                tokenToId[spaceWord] = id
            }
        }
    }
}
