
const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
  "can", "did", "do", "does", "for", "from", "had", "has", "have",
  "how", "i", "if", "in", "into", "is", "it", "its", "just", "me",
  "my", "no", "not", "of", "on", "or", "our", "so", "than", "that",
  "the", "their", "them", "then", "there", "these", "they", "this",
  "to", "too", "up", "us", "was", "we", "were", "what", "when",
  "where", "which", "who", "why", "will", "with", "would", "you",
  "your", "yours",
]);

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t && !STOPWORDS.has(t));
}

function trigramSet(text) {
  const padded = "  " + text.toLowerCase().replace(/\s+/g, " ").trim() + "  ";
  const grams = new Set();
  for (let i = 0; i < padded.length - 2; i++) grams.add(padded.slice(i, i + 3));
  return grams;
}

function jaccardSimilarity(setA, setB) {
  let intersection = 0;
  for (const gram of setA) if (setB.has(gram)) intersection++;
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

class HybridSearchIndex {
  constructor(knowledgeBase) {
    this.entries = knowledgeBase;

    this.docTokens = knowledgeBase.map((e) => tokenize(e.q + " " + e.a));
    this.docTrigrams = knowledgeBase.map((e) => trigramSet(e.q + " " + e.a));
    this.N = this.docTokens.length;

    this.documentFrequency = new Map();
    for (const tokens of this.docTokens) {
      for (const term of new Set(tokens)) {
        this.documentFrequency.set(term, (this.documentFrequency.get(term) || 0) + 1);
      }
    }

    this.avgDocLength = this.docTokens.reduce((sum, d) => sum + d.length, 0) / this.N;

    this.k1 = 1.5;
    this.b = 0.75;
  }

  inverseDocFrequency(term) {
    const n = this.documentFrequency.get(term) || 0;
    return Math.log((this.N - n + 0.5) / (n + 0.5) + 1);
  }

  bm25Score(queryTerms, docIndex) {
    const doc = this.docTokens[docIndex];
    const docLength = doc.length;
    let score = 0;
    for (const term of queryTerms) {
      const termFrequency = doc.filter((t) => t === term).length;
      if (termFrequency === 0) continue;
      const numerator = termFrequency * (this.k1 + 1);
      const denominator =
        termFrequency + this.k1 * (1 - this.b + this.b * (docLength / this.avgDocLength));
      score += this.inverseDocFrequency(term) * (numerator / denominator);
    }
    return score;
  }

  ranksFromScores(scores) {
    const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    const ranks = new Array(scores.length);
    order.forEach((docIndex, rank) => {
      ranks[docIndex] = rank;
    });
    return ranks;
  }

  search(query, topK = 1) {
    const queryTerms = tokenize(query);
    const queryTrigrams = trigramSet(query);

    const bm25Scores = this.entries.map((_, i) => this.bm25Score(queryTerms, i));
    const trigramScores = this.entries.map((_, i) => jaccardSimilarity(queryTrigrams, this.docTrigrams[i]));

    const bm25Ranks = this.ranksFromScores(bm25Scores);
    const trigramRanks = this.ranksFromScores(trigramScores);

    const RRF_K = 60; 
    const results = this.entries.map((entry, i) => ({
      entry,
      bm25Score: bm25Scores[i],
      trigramScore: trigramScores[i],
      combinedScore: 1 / (RRF_K + bm25Ranks[i] + 1) + 1 / (RRF_K + trigramRanks[i] + 1),
    }));

    results.sort((a, b) => b.combinedScore - a.combinedScore);
    return results.slice(0, topK);
  }
}