
const MIN_BM25_SCORE = 5;
const MIN_TRIGRAM_SCORE = 0.15;

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

const searchIndex = new HybridSearchIndex(KNOWLEDGE_BASE);

function addMessage(role, text) {
  const row = document.createElement("div");
  row.className = "msg-row " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble " + role;
  bubble.textContent = text;
  row.appendChild(bubble);
  chatLog.appendChild(row);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function handleSend(e) {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  addMessage("user", message);
  chatInput.value = "";

  const [best] = searchIndex.search(message, 1);
  const confident = best.bm25Score >= MIN_BM25_SCORE || best.trigramScore >= MIN_TRIGRAM_SCORE;

  if (confident) {
    addMessage("bot", best.entry.a);
  } else {
    addMessage(
      "bot",
      "I don't have a good answer for that."
    );
  }
}

chatForm.addEventListener("submit", handleSend);


addMessage(
  "bot",
  `hi, ask me anything`
);
chatInput.focus();