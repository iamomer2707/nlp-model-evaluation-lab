const state = {
  benchmark: null,
  baseline: null,
  csvText: null,
  fileName: null,
};

const $ = (selector) => document.querySelector(selector);
const percentage = (value) => `${(Number(value) * 100).toFixed(2)}%`;
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[char]));

function showToast(message) {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  document.body.append(toast);
  window.setTimeout(() => toast.remove(), 3400);
}

function setApiStatus(ready, message) {
  const chip = $('.runtime-chip');
  chip.classList.toggle('offline', !ready);
  $('#api-dot').style.background = ready ? 'var(--green)' : 'var(--amber)';
  $('#api-status').textContent = message;
}

function parameterLabel(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  return `${(value / 1_000).toFixed(0)}K`;
}

function renderBenchmark(models) {
  const sortBy = $('#sort-models').value;
  const sorted = [...models].sort((a, b) => {
    if (sortBy === 'parameters') return a.parameters - b.parameters;
    return b[sortBy] - a[sortBy];
  });
  const bestF1 = Math.max(...models.map((model) => model.f1));
  const bestAccuracy = Math.max(...models.map((model) => model.accuracy));
  $('#model-rows').innerHTML = sorted.map((model) => {
    const family = model.family === 'from_scratch' ? 'FROM SCRATCH' : 'FINE-TUNED';
    const familyClass = model.family === 'from_scratch' ? '' : ' pretrained';
    return `<tr>
      <td><span class="model-name">${model.name}</span></td>
      <td><span class="model-family${familyClass}">${family}</span></td>
      <td class="${model.accuracy === bestAccuracy ? 'score-strong' : ''}">${percentage(model.accuracy)}</td>
      <td>${percentage(model.precision)}</td>
      <td>${percentage(model.recall)}</td>
      <td class="${model.f1 === bestF1 ? 'score-strong' : ''}">${percentage(model.f1)}</td>
      <td class="param-cell">${parameterLabel(model.parameters)}</td>
    </tr>`;
  }).join('');

  const lightest = [...models].sort((a, b) => a.parameters - b.parameters)[0];
  $('#top-accuracy').innerHTML = `${(bestAccuracy * 100).toFixed(2)}<span class="unit">%</span>`;
  $('#top-f1').innerHTML = `${(bestF1 * 100).toFixed(2)}<span class="unit">%</span>`;
  $('#top-accuracy-model').textContent = models.filter((model) => model.accuracy === bestAccuracy).map((model) => model.name.toUpperCase()).join(' / ');
  $('#top-f1-model').textContent = models.filter((model) => model.f1 === bestF1).map((model) => model.name.toUpperCase()).join(' / ');
  const lightCard = document.querySelector('.highlight-card');
  lightCard.querySelector('strong').innerHTML = `${(lightest.parameters / 1_000_000).toFixed(2)}<span class="unit">M</span>`;
  lightCard.querySelector('small').textContent = `${lightest.name.toUpperCase()} PARAMETERS · ${percentage(lightest.f1)} SPAM F1`;
  lightCard.querySelector('.metric-rule i').style.width = `${Math.max(5, 100 * lightest.parameters / Math.max(...models.map((item) => item.parameters)))}%`;
}

function updateBaseline(metrics) {
  state.baseline = metrics;
  $('#dataset-count').textContent = Number(metrics.messages).toLocaleString();
  setApiStatus(true, 'DEMO READY');
}

function renderPrediction(data) {
  const isPositive = data.prediction === data.positive_label;
  $('#empty-result').classList.add('hidden');
  $('#prediction-error').classList.add('hidden');
  $('#result-content').classList.remove('hidden');
  const resultLabel = $('#result-label');
  resultLabel.textContent = data.prediction;
  resultLabel.classList.toggle('ham', !isPositive);
  $('#result-icon').classList.toggle('ham', !isPositive);
  $('#result-icon').textContent = isPositive ? '◆' : '◇';
  $('#result-confidence').textContent = percentage(data.confidence);
  $('#spam-probability').textContent = percentage(data.positive_probability);
  $('#probability-fill').style.width = `${Math.max(1, data.positive_probability * 100)}%`;
  $('#inference-time').textContent = `${Number(data.latency_ms).toFixed(1)} MS`;
}

function renderEvaluation(data) {
  $('#eval-empty').classList.add('hidden');
  $('#eval-content').classList.remove('hidden');
  $('#eval-title').textContent = 'Holdout results';
  const [negativeRaw, positiveRaw] = data.class_order;
  const negative = escapeHtml(negativeRaw);
  const positive = escapeHtml(positiveRaw);
  const positiveLabel = escapeHtml(data.positive_label);
  const matrix = data.confusion_matrix;
  $('#eval-content').innerHTML = `
    <div class="eval-grid">
      <div class="eval-metric"><small>ACCURACY</small><strong>${percentage(data.accuracy)}</strong></div>
      <div class="eval-metric"><small>PRECISION*</small><strong>${percentage(data.precision)}</strong></div>
      <div class="eval-metric"><small>RECALL*</small><strong>${percentage(data.recall)}</strong></div>
      <div class="eval-metric"><small>F1*</small><strong>${percentage(data.f1)}</strong></div>
    </div>
    <p class="eval-summary">${Number(data.rows).toLocaleString()} rows · ${Number(data.train_rows).toLocaleString()} train · ${Number(data.test_rows).toLocaleString()} test · positive class: <b>${positiveLabel}</b></p>
    <div class="confusion-wrap"><table class="confusion-table" aria-label="Confusion matrix"><thead><tr><th>ACTUAL ↓ / PREDICTED →</th><th>${negative}</th><th>${positive}</th></tr></thead><tbody><tr><th>${negative}</th><td>${matrix[0][0]}</td><td>${matrix[0][1]}</td></tr><tr><th>${positive}</th><td>${matrix[1][0]}</td><td>${matrix[1][1]}</td></tr></tbody></table></div>
    <p class="fine-print">*Precision, recall, and F1 use “${positiveLabel}” as the positive class. The CSV stays in memory and is discarded after evaluation.</p>`;
}

function showError(target, message) {
  target.textContent = message;
  target.classList.remove('hidden');
}

async function loadOverview() {
  try {
    const response = await fetch('/api/overview');
    if (!response.ok) throw new Error('Could not load the experiment summary.');
    const data = await response.json();
    state.benchmark = data.benchmark;
    renderBenchmark(data.benchmark.models);
    if (data.demo_ready) {
      updateBaseline(data.demo_metrics);
    } else {
      setApiStatus(false, 'DEMO OFFLINE');
      const error = $('#prediction-error');
      showError(error, data.demo_error || 'Live classification is unavailable. The benchmark dashboard is still ready.');
    }
  } catch (error) {
    setApiStatus(false, 'API UNAVAILABLE');
    $('#model-rows').innerHTML = '<tr><td colspan="7" class="loading-row">BENCHMARK DATA COULD NOT BE LOADED</td></tr>';
    showToast(error.message || 'Could not connect to the local API.');
  }
}

$('#message-input').addEventListener('input', (event) => {
  $('#char-count').textContent = `${event.target.value.length.toLocaleString()} / 3,000`;
});

document.querySelectorAll('[data-sample]').forEach((button) => {
  button.addEventListener('click', () => {
    const input = $('#message-input');
    input.value = button.dataset.sample;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.focus();
  });
});

$('#classify-button').addEventListener('click', async () => {
  const message = $('#message-input').value.trim();
  const error = $('#prediction-error');
  error.classList.add('hidden');
  if (!message) {
    showError(error, 'Add a message or load one of the samples to run the classifier.');
    return;
  }
  const button = $('#classify-button');
  button.disabled = true;
  button.querySelector('span:first-child').textContent = 'CLASSIFYING…';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Prediction failed.');
    renderPrediction(data);
  } catch (err) {
    showError(error, err.message || 'The classifier could not process this message.');
  } finally {
    button.disabled = false;
    button.querySelector('span:first-child').textContent = 'RUN CLASSIFIER';
  }
});

$('#sort-models').addEventListener('change', () => {
  if (state.benchmark) renderBenchmark(state.benchmark.models);
});

$('#csv-input').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  const error = $('#upload-error');
  error.classList.add('hidden');
  state.csvText = null;
  state.fileName = null;
  $('#evaluate-button').disabled = true;
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.csv')) {
    showError(error, 'Choose a .csv file.');
    return;
  }
  if (file.size > 2_000_000) {
    showError(error, 'This file is larger than 2 MB.');
    return;
  }
  try {
    const content = await file.text();
    if (!content.trim()) throw new Error('The selected CSV is empty.');
    state.csvText = content;
    state.fileName = file.name;
    $('#evaluate-button').disabled = false;
    $('.file-drop').classList.add('has-file');
    $('.file-drop strong').textContent = file.name;
    $('.file-drop small').textContent = `${Math.max(1, Math.round(file.size / 1024))} KB · ready to evaluate`;
  } catch (err) {
    showError(error, err.message || 'Could not read this file.');
  }
});

$('#evaluate-button').addEventListener('click', async () => {
  if (!state.csvText) return;
  const button = $('#evaluate-button');
  const error = $('#upload-error');
  error.classList.add('hidden');
  button.disabled = true;
  button.innerHTML = 'FITTING BASELINE… <span>◌</span>';
  try {
    const response = await fetch('/api/evaluate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csv_text: state.csvText, filename: state.fileName }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Evaluation failed.');
    renderEvaluation(data);
    $('#evaluation-result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    showError(error, err.message || 'Could not evaluate this dataset.');
  } finally {
    button.disabled = !state.csvText;
    button.innerHTML = 'RUN HOLDOUT EVALUATION <span>→</span>';
  }
});

document.querySelectorAll('.nav-link').forEach((link) => {
  link.addEventListener('click', () => {
    document.querySelectorAll('.nav-link').forEach((item) => item.classList.remove('active'));
    link.classList.add('active');
  });
});

loadOverview();
