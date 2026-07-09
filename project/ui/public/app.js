// ─── Config ───
let API_BASE = "http://localhost:8010";

async function loadConfig() {
  try {
    const resp = await fetch("/config");
    const cfg = await resp.json();
    API_BASE = cfg.apiBase;
  } catch {
    // fallback
  }
}

// ─── i18n ───
const I18N = {
  ru: {
    title: "MailKB — управление email-архивом",
    tab_pipeline: "🔧 Pipeline",
    tab_search: "🔍 Поиск",
    pipeline_title: "Pipeline обработки писем",
    pipeline_subtitle: "Выполняйте шаги по порядку или запустите всё сразу.",
    run_all: "🚀 Запустить весь pipeline",
    search_title: "Поиск и анализ по проекту",
    search_subtitle: "Найдите информацию в email-архиве по названию проекта.",
    search_placeholder: "Название проекта, например: segezha, sibur",
    btn_threads: "🔍 Поиск тредов",
    btn_corpus: "📋 Корпус батч",
    btn_batch: "📊 Батч-анализ",
    btn_global: "🌐 Глобальный отчёт",
    btn_clear: "🗑 Очистить summaries",
    batch_limit: "Лимит батчей:",
    step_1: "1. Создание таблиц БД",
    step_2: "2. Загрузка писем из mbox",
    step_3: "3. Дедупликация писем",
    step_4: "4. Очистка тел писем (LLM)",
    step_5: "5. Парсинг писем (LLM)",
    step_6: "6. Индексация в Qdrant",
    status_idle: "⏳ ожидание",
    status_running: "выполняется…",
    status_ok: "✅ OK",
    status_error: "❌ Ошибка",
    params_title: "Параметры",
    params_apply: "Применить",
    params_cancel: "Отмена",
    params_saved: "✅ Параметры сохранены",
    enter_project: "⚠️ Введите название проекта",
    no_results: "По запросу «{q}» ничего не найдено",
    corpus_empty: "Корпус пуст. Найдено тем: {n}",
    corpus_info: "📊 Всего сообщений: {total}, показано: {shown}{more}",
    has_more: " (есть ещё)",
    summaries_cleared: "🧹 Summaries очищены",
    pipeline_done: "✅ Pipeline завершён",
    run_all_running: "⏳ Выполняется…",
  },
  en: {
    title: "MailKB — Email Archive Management",
    tab_pipeline: "🔧 Pipeline",
    tab_search: "🔍 Search",
    pipeline_title: "Email Processing Pipeline",
    pipeline_subtitle: "Execute steps in order or run all at once.",
    run_all: "🚀 Run Full Pipeline",
    search_title: "Project Search & Analysis",
    search_subtitle: "Find information in the email archive by project name.",
    search_placeholder: "Project name, e.g. segezha, sibur",
    btn_threads: "🔍 Search Threads",
    btn_corpus: "📋 Corpus Batch",
    btn_batch: "📊 Batch Analysis",
    btn_global: "🌐 Global Report",
    btn_clear: "🗑 Clear Summaries",
    batch_limit: "Batch limit:",
    step_1: "1. Create DB Tables",
    step_2: "2. Import Emails from mbox",
    step_3: "3. Deduplicate Emails",
    step_4: "4. Clean Email Bodies (LLM)",
    step_5: "5. Parse Emails (LLM)",
    step_6: "6. Index into Qdrant",
    status_idle: "⏳ waiting",
    status_running: "running…",
    status_ok: "✅ OK",
    status_error: "❌ Error",
    params_title: "Parameters",
    params_apply: "Apply",
    params_cancel: "Cancel",
    params_saved: "✅ Parameters saved",
    enter_project: "⚠️ Enter a project name",
    no_results: "No results for «{q}»",
    corpus_empty: "Corpus is empty. Topics found: {n}",
    corpus_info: "📊 Total messages: {total}, shown: {shown}{more}",
    has_more: " (more)",
    summaries_cleared: "🧹 Summaries cleared",
    pipeline_done: "✅ Pipeline complete",
    run_all_running: "⏳ Running…",
  }
};

let currentLang = localStorage.getItem("mailkb_lang") || "ru";

function t(key, params) {
  let s = (I18N[currentLang] || I18N.ru)[key];
  if (s === undefined) s = key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      s = s.replace(`{${k}}`, v);
    }
  }
  return s;
}

function applyLanguage() {
  document.documentElement.lang = currentLang;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  document.getElementById("lang-btn").textContent = currentLang.toUpperCase();
  // Re-render pipeline labels
  renderPipeline();
}

function toggleLang() {
  currentLang = currentLang === "ru" ? "en" : "ru";
  localStorage.setItem("mailkb_lang", currentLang);
  applyLanguage();
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
  { key: "init-db",          labelKey: "step_1", icon: "🗄️", hasParams: false },
  { key: "import-mbox",      labelKey: "step_2", icon: "📥", hasParams: false },
  { key: "dedup",            labelKey: "step_3", icon: "🧹", hasParams: false },
  { key: "clean-bodies",     labelKey: "step_4", icon: "✨", hasParams: true },
  { key: "parse",            labelKey: "step_5", icon: "📄", hasParams: true },
  { key: "index-messages",   labelKey: "step_6", icon: "🔢", hasParams: true },
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
      <span class="step-label">${t(step.labelKey)}</span>
      <span class="step-status idle" id="status-${step.key}">${t("status_idle")}</span>
      <span class="step-detail" id="detail-${step.key}"></span>
      ${paramsHtml}
      <button class="btn btn-primary run-btn" data-key="${step.key}">▶ ${currentLang === "en" ? "Run" : "Запуск"}</button>
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
        <h3>${t("params_title")}: ${key}</h3>
        <div class="modal-fields">${fields}</div>
        <div class="modal-actions">
          <button class="btn btn-primary" id="params-apply">${t("params_apply")}</button>
          <button class="btn btn-search" id="params-cancel">${t("params_cancel")}</button>
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
    showToast(t("params_saved"), "success");
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
  statusEl.innerHTML = `<span class="spinner"></span> ${t("status_running")}`;
  btn.disabled = true;

  try {
    let body;
    if (PIPELINE_STEPS.find((s) => s.key === key).hasParams) {
      body = currentParams[key];
    }

    const result = await callApi("POST", "/pipeline/" + key, body);

    statusEl.className = "step-status ok";
    statusEl.textContent = t("status_ok");

    if (key === "init-db" && result.results) {
      const errs = result.results.filter((r) => r.status !== "ok");
      if (errs.length) {
        statusEl.className = "step-status error";
        statusEl.textContent = "⚠️ " + errs.length + " " + (currentLang === "en" ? "errors" : "ошибок");
        detailEl.textContent = errs.map((e) => e.file + ": " + e.status).join("; ");
      } else {
        detailEl.textContent = result.results.map((r) => r.file).join(", ");
      }
    }
    if (key === "parse" && result.result) {
      detailEl.textContent = `success=${result.result.success_count}, errors=${result.result.error_count}`;
    }

    showToast(`✅ ${t(PIPELINE_STEPS.find((s) => s.key === key).labelKey)} — OK`, "success");
  } catch (err) {
    statusEl.className = "step-status error";
    statusEl.textContent = t("status_error");
    detailEl.textContent = err.message;
    showToast(`❌ ${t(PIPELINE_STEPS.find((s) => s.key === key).labelKey)}: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

// ─── Run all ───
async function runAll() {
  const btn = document.getElementById("run-all");
  btn.disabled = true;
  btn.textContent = t("run_all_running");

  for (const step of PIPELINE_STEPS) {
    await runSingle(step.key);
  }

  btn.disabled = false;
  btn.textContent = t("run_all");
  showToast(t("pipeline_done"), "success");
}

// ─── Search ───
async function handleSearch(action) {
  const input = document.getElementById("project-input");
  const projectHint = input.value.trim();
  if (!projectHint && action !== "clear-summaries") {
    showToast(t("enter_project"), "error");
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
      const maxBatches = parseInt(document.getElementById("max-batches").value) || 0;
      body = { project_hint: projectHint, max_batches: maxBatches };
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
      resultsArea.innerHTML = `<div class="result-card success">${t("summaries_cleared")}</div>`;
      showToast(t("summaries_cleared"), "success");
      return;
    }

    let html;
    if (action === "search-threads") {
      html = formatThreads(result);
    } else if (action === "corpus-batch") {
      html = formatCorpusBatch(result);
    } else if (action === "batch-analysis" || action === "global-analysis") {
      const text = result.result || JSON.stringify(result, null, 2);
      html = formatAnalysisMarkdown(text);
    }
    resultsArea.innerHTML = html;
  } catch (err) {
    resultsArea.innerHTML = `<div class="result-card error">❌ ${escapeHtml(err.message)}</div>`;
    showToast(`❌ ${err.message}`, "error");
  }
}

function formatThreads(data) {
  const threads = data.threads || [];
  if (!threads.length) return `<div class="result-card info">${t("no_results", {q: escapeHtml(data.project_hint)})}</div>`;

  return threads.map(t => `
    <div class="result-card thread-card">
      <div class="thread-header">
        <span class="thread-key">${escapeHtml(t.thread_key || t.topic || '')}</span>
        <span class="thread-date">${escapeHtml(t.date || '')}</span>
      </div>
      <div class="thread-subject">${escapeHtml(t.subject || '')}</div>
      ${t.keywords && t.keywords.length ? `<div class="thread-keywords">🏷️ ${t.keywords.map(k => `<span class="keyword">${escapeHtml(k)}</span>`).join(' ')}</div>` : ''}
      <div class="thread-meta">
        <span>👥 ${(t.participants || []).map(p => escapeHtml(p)).join(', ') || '—'}</span>
      </div>
      ${t.snippet ? `<div class="thread-snippet">${escapeHtml(t.snippet)}</div>` : ''}
    </div>
  `).join('');
}

function formatCorpusBatch(data) {
  const batch = data.batch || [];
  if (!batch.length) return `<div class="result-card info">${t("corpus_empty", {n: data.thread_keys ? data.thread_keys.length : 0})}</div>`;

  const more = data.has_more ? t("has_more") : "";
  let html = `<div class="result-card info">${t("corpus_info", {total: data.total_messages, shown: batch.length, more})}</div>`;
  html += batch.map(m => `
    <div class="result-card msg-card">
      <div class="msg-header">
        <span class="msg-topic">${escapeHtml(m.topic || m.subject || '')}</span>
        <span class="msg-date">${escapeHtml(m.sent_at_utc || '')}</span>
      </div>
      <div class="msg-meta">
        <span>📧 ${escapeHtml(m.email_id || '')}</span>
        <span>👥 ${(m.participants || []).map(p => escapeHtml(p)).join(', ') || '—'}</span>
      </div>
      ${m.snippet ? `<div class="msg-snippet">${escapeHtml(m.snippet)}</div>` : ''}
    </div>
  `).join('');
  return html;
}

function formatAnalysisMarkdown(text) {
  let html;
  try {
    html = marked.parse(text, { breaks: true, gfm: true });
  } catch {
    html = `<pre>${escapeHtml(text)}</pre>`;
  }
  return `<div class="markdown-body">${html}</div>`;
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

  // Language
  document.getElementById("lang-btn").addEventListener("click", toggleLang);
  applyLanguage();

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
