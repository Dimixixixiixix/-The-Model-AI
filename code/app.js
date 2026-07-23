

const HIDDEN_SIZE = 96;
const ITERATIONS = 40000;
const LEARNING_RATE = 0.1;
const BATCH_PER_FRAME = 5; 

let model = null;

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const trainingBanner = document.getElementById("trainingBanner");
const trainingProgress = document.getElementById("trainingProgress");

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

function startTraining() {
  model = new CharRNN(TRAINING_TEXT, HIDDEN_SIZE);

  let it = 0;

  function step() {
    if (it >= ITERATIONS) {
      finishTraining();
      return;
    }
    for (let b = 0; b < BATCH_PER_FRAME && it < ITERATIONS; b++, it++) {
      model.trainStep(LEARNING_RATE);
    }
    const pct = Math.round((it / ITERATIONS) * 100);
    trainingProgress.style.width = pct + "%";
    trainingBanner.querySelector(".training-text").textContent =
      `Training the model… ${pct}% (loss: ${model.smoothLoss.toFixed(2)})`;
    requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function finishTraining() {
  trainingBanner.style.display = "none";
  chatInput.disabled = false;
  sendBtn.disabled = false;
  chatInput.placeholder = "Type a message...";
  addMessage("bot", "Hi! I'm a tiny neural network trained on a small set of examples. Ask me something simple, like \"what is your name?\" or \"who made you?\"");
  chatInput.focus();
}

function handleSend(e) {
  e.preventDefault();
  if (!model) return;
  const message = chatInput.value.trim();
  if (!message) return;

  addMessage("user", message);
  chatInput.value = "";

  const { text } = model.answer(message);
  addMessage("bot", text || "(I couldn't come up with an answer for that — try asking something closer to my training examples.)");
}

chatForm.addEventListener("submit", handleSend);

startTraining();