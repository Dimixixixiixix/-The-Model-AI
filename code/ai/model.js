// NOTE: this shi is old and deprecated, this was for the old char-rnn model :/


const SEQ_LENGTH = 25; // note for dimix from the future: dont change this or you will break it :^
const STOP_MARKERS = ["\nQ", "\nA", "\n\n"]; 



function gaussianRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function randMat(rows, cols, scale) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    for (let j = 0; j < cols; j++) row[j] = gaussianRandom() * scale;
    m.push(row);
  }
  return m;
}

function zerosMat(rows, cols) {
  const m = [];
  for (let i = 0; i < rows; i++) m.push(new Float64Array(cols));
  return m;
}

function matVec(W, x) {
  const out = new Float64Array(W.length);
  for (let i = 0; i < W.length; i++) {
    let s = 0;
    const row = W[i];
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

function matVecTranspose(W, x) {
  const cols = W[0].length;
  const out = new Float64Array(cols);
  for (let i = 0; i < W.length; i++) {
    const row = W[i];
    const xi = x[i];
    if (xi === 0) continue;
    for (let j = 0; j < cols; j++) out[j] += row[j] * xi;
  }
  return out;
}

function addOuterInPlace(M, a, b) {
  for (let i = 0; i < a.length; i++) {
    const row = M[i];
    const ai = a[i];
    if (ai === 0) continue;
    for (let j = 0; j < b.length; j++) row[j] += ai * b[j];
  }
}

function tanhVec(x) {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.tanh(x[i]);
  return out;
}

function softmax(x) {
  let max = -Infinity;
  for (let i = 0; i < x.length; i++) if (x[i] > max) max = x[i];
  const out = new Float64Array(x.length);
  let sum = 0;
  for (let i = 0; i < x.length; i++) { out[i] = Math.exp(x[i] - max); sum += out[i]; }
  for (let i = 0; i < x.length; i++) out[i] /= sum;
  return out;
}

function clip(v, lo, hi) {
  for (let i = 0; i < v.length; i++) v[i] = Math.max(lo, Math.min(hi, v[i]));
}

function clipMat(M, lo, hi) {
  for (const row of M) clip(row, lo, hi);
}

function addAll(...vecs) {
  const out = new Float64Array(vecs[0].length);
  for (const v of vecs) for (let i = 0; i < v.length; i++) out[i] += v[i];
  return out;
}

function adagradUpdate(param, grad, mem, lr) {
  for (let i = 0; i < param.length; i++) {
    const p = param[i], g = grad[i], m = mem[i];
    for (let j = 0; j < p.length; j++) {
      m[j] += g[j] * g[j];
      p[j] += (-lr * g[j]) / Math.sqrt(m[j] + 1e-8);
    }
  }
}

function adagradUpdateVec(param, grad, mem, lr) {
  for (let i = 0; i < param.length; i++) {
    mem[i] += grad[i] * grad[i];
    param[i] += (-lr * grad[i]) / Math.sqrt(mem[i] + 1e-8);
  }
}

function sampleFromDistribution(p) {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < p.length; i++) {
    cum += p[i];
    if (r <= cum) return i;
  }
  return p.length - 1;
}

// insert cool comment here (code for model is below)

class CharRNN {
  constructor(text, hiddenSize) {
    this.text = text;
    this.chars = Array.from(new Set(text)).sort();
    this.vocabSize = this.chars.length;
    this.charToIx = {};
    this.chars.forEach((c, i) => (this.charToIx[c] = i));
    this.hiddenSize = hiddenSize;

   
    this.Wxh = randMat(hiddenSize, this.vocabSize, 0.01); 
    this.Whh = randMat(hiddenSize, hiddenSize, 0.01);      
    this.Why = randMat(this.vocabSize, hiddenSize, 0.01);  
    this.bh = new Float64Array(hiddenSize);
    this.by = new Float64Array(this.vocabSize);

    
    this.mWxh = zerosMat(hiddenSize, this.vocabSize);
    this.mWhh = zerosMat(hiddenSize, hiddenSize);
    this.mWhy = zerosMat(this.vocabSize, hiddenSize);
    this.mbh = new Float64Array(hiddenSize);
    this.mby = new Float64Array(this.vocabSize);

    this.hprev = new Float64Array(hiddenSize);
    this.dataPtr = 0;
    this.smoothLoss = -Math.log(1 / this.vocabSize) * SEQ_LENGTH;
  }

  onehot(ix) {
    const v = new Float64Array(this.vocabSize);
    v[ix] = 1;
    return v;
  }


  trainStep(learningRate) {
    const seqLen = SEQ_LENGTH;
    if (this.dataPtr + seqLen + 1 >= this.text.length) {
      this.hprev = new Float64Array(this.hiddenSize);
      this.dataPtr = 0;
    }

    const inputs = [];
    const targets = [];
    for (let t = 0; t < seqLen; t++) {
      inputs.push(this.charToIx[this.text[this.dataPtr + t]]);
      targets.push(this.charToIx[this.text[this.dataPtr + t + 1]]);
    }


    const xs = [], hs = [this.hprev], ps = [];
    let loss = 0;
    for (let t = 0; t < seqLen; t++) {
      const x = this.onehot(inputs[t]);
      xs.push(x);
      const h = tanhVec(addAll(matVec(this.Wxh, x), matVec(this.Whh, hs[t]), this.bh));
      hs.push(h);
      const y = addAll(matVec(this.Why, h), this.by);
      const p = softmax(y);
      ps.push(p);
      loss += -Math.log(Math.max(p[targets[t]], 1e-12));
    }


    const dWxh = zerosMat(this.hiddenSize, this.vocabSize);
    const dWhh = zerosMat(this.hiddenSize, this.hiddenSize);
    const dWhy = zerosMat(this.vocabSize, this.hiddenSize);
    const dbh = new Float64Array(this.hiddenSize);
    const dby = new Float64Array(this.vocabSize);
    let dhnext = new Float64Array(this.hiddenSize);

    for (let t = seqLen - 1; t >= 0; t--) {
      const dy = ps[t].slice();
      dy[targets[t]] -= 1;
      addOuterInPlace(dWhy, dy, hs[t + 1]);
      for (let i = 0; i < dby.length; i++) dby[i] += dy[i];

      const dh = addAll(matVecTranspose(this.Why, dy), dhnext);
      const hCur = hs[t + 1];
      const dhraw = new Float64Array(this.hiddenSize);
      for (let i = 0; i < this.hiddenSize; i++) dhraw[i] = (1 - hCur[i] * hCur[i]) * dh[i];

      for (let i = 0; i < this.hiddenSize; i++) dbh[i] += dhraw[i];
      addOuterInPlace(dWxh, dhraw, xs[t]);
      addOuterInPlace(dWhh, dhraw, hs[t]);
      dhnext = matVecTranspose(this.Whh, dhraw);
    }

    clipMat(dWxh, -5, 5); clipMat(dWhh, -5, 5); clipMat(dWhy, -5, 5);
    clip(dbh, -5, 5); clip(dby, -5, 5);

    adagradUpdate(this.Wxh, dWxh, this.mWxh, learningRate);
    adagradUpdate(this.Whh, dWhh, this.mWhh, learningRate);
    adagradUpdate(this.Why, dWhy, this.mWhy, learningRate);
    adagradUpdateVec(this.bh, dbh, this.mbh, learningRate);
    adagradUpdateVec(this.by, dby, this.mby, learningRate);

    this.hprev = hs[seqLen];
    this.dataPtr += seqLen;
    this.smoothLoss = this.smoothLoss * 0.999 + loss * 0.001;
    return this.smoothLoss;
  }


  primeState(text) {
    let h = new Float64Array(this.hiddenSize);
    let lastIx = 0;
    const unknown = [];
    for (const ch of text) {
      if (!(ch in this.charToIx)) {
        unknown.push(ch);
        continue; 
      }
      const ix = this.charToIx[ch];
      const x = this.onehot(ix);
      h = tanhVec(addAll(matVec(this.Wxh, x), matVec(this.Whh, h), this.bh));
      lastIx = ix;
    }
    return { h, lastIx, unknown };
  }


  generateFrom(h, seedIx, maxLen, stopMarkers) {
    let out = "";
    let ix = seedIx;
    for (let t = 0; t < maxLen; t++) {
      const x = this.onehot(ix);
      h = tanhVec(addAll(matVec(this.Wxh, x), matVec(this.Whh, h), this.bh));
      const y = addAll(matVec(this.Why, h), this.by);
      const p = softmax(y);
      ix = sampleFromDistribution(p);
      out += this.chars[ix];
      for (const marker of stopMarkers) {
        const idx = out.indexOf(marker);
        if (idx !== -1) return out.slice(0, idx);
      }
    }
    return out; 
  }


  answer(question) {
    const primeText = "Q: " + question + "\nA:";
    const { h, lastIx, unknown } = this.primeState(primeText);
    const raw = this.generateFrom(h, lastIx, 250, STOP_MARKERS);
    return { text: raw.trim(), unknown };
  }
}