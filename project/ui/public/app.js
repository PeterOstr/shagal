// ─── Config ───
let API_BASE = "http://localhost:8010";

async function loadConfig() {
  try {
    const resp = await fetch("/config");
    const cfg = await resp.json();
    API_BASE = cfg.apiBase;
  } catch {
    // fallback to default
  }
}

// ─── API call helper ───
async function callApi(method, path, body) {
  const url = API_BASE + path;
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body !== undefined) {
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return resp.json();
}

// ─── Toast ───
function showToast(msg, type) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "toast " + (type || "");
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 4000);
}

// ─── Pipeline steps definition ───
const PIPELINE_STEPS = [
  { key: "init-db",          label: "1. Создание таблиц БД",           icon: "🗄️", hasParams: false },
  { key: "import-mbox",      label: "2. Загрузка писем из mbox",       icon: "📥", hasParams: false },
  { key: "dedup",            label: "3. Дедупликация писем",            icon: "🧹", hasParams: false },
  { key: "clean-bodies",     label: "4. Очистка тел писем (LLM)",       icon: "✨", hasParams: true },
  { key: "parse",            label: "5. Парсинг писем (LLM)",           icon: "📄", hasParams: true },
  { key: "index-messages",   label: "6. Индексация в Qdrant",           icon: "🔢", hasParams: true },
];

function renderPipeline() {
  const container = document.getElementById("pipeline-steps");
  container.innerHTML = "";

  PIPELINE_STEPS.forEach((step) => {
    const div = document.createElement("div");
    div.className = "step";
    div.id = "step-" + step.key;

    let paramsHtml = "";
    if (step.hasParams) {
      paramsHtml = `<button class="btn btn-search params-btn" data-key="${step.key}">⚙️</button>`;
    }

    div.innerHTML = `
      <span class="step-icon">${step.icon}</span>
      <span class="step-label">${step.label}</span>
      <span class="step-status idle" id="status-${step.key}">⏳ ожидание</span>
      <span class="step-detail" id="detail-${step.key}"></span>
      ${paramsHtml}
      <button class="btn btn-primary run-btn" data-key="${step.key}">▶ Запуск</button>
    `;

    container.appendChild(div);
  });

  // Event delegation
  container.addEventListener("click", (e) => {
    const runBtn = e.target.closest(".run-btn");
    if (runBtn) {
      runSingle(runBtn.dataset.key);
      return;
    }
    const paramsBtn = e.target.closest(".params-btn");
    if (paramsBtn) {
      showParamsModal(paramsBtn.dataset.key);
    }
  });

  document.getElementById("run-all").addEventListener("click", runAll);
}

// ─── Params modal ───
const PARAMS_DEFAULTS = {
  "clean-bodies": { fetch_batch: 30, llm_batch: 5 },
  "parse":        { limit: 50, batch_size: 3, max_workers: 6 },
  "index-messages": { batch_size: 1000, recreate: false },
};

let currentParams = JSON.parse(JSON.stringify(PARAMS_DEFAULTS));

function showParamsModal(key) {
  const vals = currentParams[key];
  const fields = Object.entries(vals)
    .map(([k, v]) => {
      const type = typeof v === "boolean" ? "checkbox" : "number";
      const checked = typeof v === "boolean" && v ? "checked" : "";
      const val = typeof v === "boolean" ? "" : v;
      return `<label>${k}: <input type="${type}" name="${k}" value="${val}" ${checked} step="1" /></label>`;
    })
    .join("");

  const html = `
    <div class="modal-overlay" id="params-modal">
      <div class="modal">
        <h3>Параметры: ${key}</h3>
        <div class="modal-fields">${fields}</div>
        <div class="modal-actions">
          <button class="btn btn-primary" id="params-apply">Применить</button>
          <button class="btn btn-search" id="params-cancel">Отмена</button>
        </div>
      </div>
    </div>
  `;

  const existing = document.getElementById("params-modal");
  if (existing) existing.remove();

  document.body.insertAdjacentHTML("beforeend", html);

  document.getElementById("params-apply").addEventListener("click", () => {
    const form = document.querySelector("#params-modal .modal-fields");
    const inputs = form.querySelectorAll("input");
    inputs.forEach((inp) => {
      if (inp.type === "checkbox") {
        vals[inp.name] = inp.checked;
      } else {
        vals[inp.name] = Number(inp.value);
      }
    });
    showToast("✅ Параметры сохранены", "success");
    document.getElementById("params-modal").remove();
  });

  document.getElementById("params-cancel").addEventListener("click", () => {
    document.getElementById("params-modal").remove();
  });
}

// ─── Single step run ───
async function runSingle(key) {
  const statusEl = document.getElementById("status-" + key);
  const detailEl = document.getElementById("detail-" + key);
  const btn = document.querySelector(`.run-btn[data-key="${key}"]`);

  statusEl.className = "step-status running";
  statusEl.innerHTML = '<span class="spinner"></span> выполняется…';
  btn.disabled = true;

  try {
    let body;
    if (PIPELINE_STEPS.find((s) => s.key === key).hasParams) {
      body = currentParams[key];
    }

    const result = await callApi("POST", "/pipeline/" + key, body);

    statusEl.className = "step-status ok";
    statusEl.textContent = "✅ OK";

    if (key === "init-db" && result.results) {
      const errs = result.results.filter((r) => r.status !== "ok");
      if (errs.length) {
        statusEl.className = "step-status error";
        statusEl.textContent = "⚠️ " + errs.length + " ошибок";
        detailEl.textContent = errs.map((e) => e.file + ": " + e.status).join("; ");
      } else {
        detailEl.textContent = result.results.map((r) => r.file).join(", ");
      }
    }
    if (key === "parse" && result.result) {
      detailEl.textContent = `success=${result.result.success_count}, errors=${result.result.error_count}`;
    }

    showToast(`✅ ${PIPELINE_STEPS.find((s) => s.key === key).label} — OK`, "success");
  } catch (err) {
    statusEl.className = "step-status error";
    statusEl.textContent = "❌ Ошибка";
    detailEl.textContent = err.message;
    showToast(`❌ ${PIPELINE_STEPS.find((s) => s.key === key).label}: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

// ─── Run all ───
async function runAll() {
  const btn = document.getElementById("run-all");
  btn.disabled = true;
  btn.textContent = "⏳ Выполняется…";

  for (const step of PIPELINE_STEPS) {
    await runSingle(step.key);
  }

  btn.disabled = false;
  btn.textContent = "🚀 Запустить весь pipeline";
  showToast("✅ Pipeline завершён", "success");
}

// ─── Search ───
async function handleSearch(action) {
  const input = document.getElementById("project-input");
  const projectHint = input.value.trim();
  if (!projectHint) {
    showToast("⚠️ Введите название проекта", "error");
    return;
  }

  const resultsArea = document.getElementById("search-results");

  let path, body;
  switch (action) {
    case "search-threads":
      path = "/search/threads";
      body = { project_hint: projectHint, limit: 10 };
      break;
    case "corpus-batch":
      path = "/search/corpus-batch";
      body = { project_hint: projectHint, batch_size: 50 };
      break;
    case "batch-analysis":
      path = "/analysis/batch";
      body = { project_hint: projectHint };
      break;
    case "global-analysis":
      path = "/analysis/global";
      body = { project_hint: projectHint };
      break;
    case "clear-summaries":
      path = "/summaries/clear";
      body = undefined;
      break;
    default:
      return;
  }

  resultsArea.innerHTML = '<div class="spinner"></div>';

  try {
    const result = await callApi("POST", path, body);

    if (action === "clear-summaries") {
      resultsArea.innerHTML = `<div class="markdown-result">🧹 Summaries очищены</div>`;
      showToast("🧹 Summaries очищены", "success");
      return;
    }

    if (action === "batch-analysis" || action === "global-analysis") {
      const text = result.result || JSON.stringify(result, null, 2);
      resultsArea.innerHTML = `<div class="markdown-result">${escapeHtml(text)}</div>`;
    } else {
      resultsArea.innerHTML = `<pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
    }
  } catch (err) {
    resultsArea.innerHTML = `<div class="markdown-result" style="color:var(--error)">❌ ${escapeHtml(err.message)}</div>`;
    showToast(`❌ ${err.message}`, "error");
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ─── Tab switching ───
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add("active");
  document.getElementById("tab-" + name).classList.add("active");
}

// ─── Init ───
(async function init() {
  await loadConfig();
  renderPipeline();

  // Tab switching
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  // Search buttons
  document.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => handleSearch(btn.dataset.action));
  });

  // Enter key in search input
  document.getElementById("project-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      handleSearch("search-threads");
    }
  });
})();
