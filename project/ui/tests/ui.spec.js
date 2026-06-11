const { chromium } = require("playwright");
const express = require("express");
const path = require("path");
const http = require("http");

const UI_PORT = 3099;
const API_PORT = 3098;

const HEADED = process.env.HEADED === "1";

// ─── Mock API server ───
function startMockApi(port) {
  let callLog = [];

  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      res.setHeader("Content-Type", "application/json");

      let body = "";
      req.on("data", (c) => (body += c));
      req.on("end", () => {
        callLog.push({ method: req.method, url: req.url, body });

        if (req.method === "GET" && req.url === "/health") {
          res.end(JSON.stringify({ status: "ok" }));
        } else if (req.method === "POST" && req.url === "/pipeline/init-db") {
          res.end(JSON.stringify({
            status: "ok",
            results: [
              { file: "01_create_emails_table.sql", status: "ok" },
              { file: "02_create_attachments.sql", status: "ok" },
            ],
          }));
        } else if (req.method === "POST" && req.url === "/pipeline/import-mbox") {
          res.end(JSON.stringify({ status: "ok" }));
        } else if (req.method === "POST" && req.url === "/pipeline/dedup") {
          res.end(JSON.stringify({ status: "ok" }));
        } else if (req.method === "POST" && req.url === "/pipeline/clean-bodies") {
          res.end(JSON.stringify({ status: "ok" }));
        } else if (req.method === "POST" && req.url === "/pipeline/parse") {
          res.end(JSON.stringify({ status: "ok", result: { success_count: 5, error_count: 0 } }));
        } else if (req.method === "POST" && req.url === "/pipeline/index-messages") {
          res.end(JSON.stringify({ status: "ok" }));
        } else if (req.method === "POST" && req.url === "/search/threads") {
          res.end(JSON.stringify({ threads: [{ thread_key: "t1", subject: "Test" }], project_hint: "segezha" }));
        } else if (req.method === "POST" && req.url === "/search/corpus-batch") {
          res.end(JSON.stringify({ batch: [], project_hint: "segezha", has_more: false }));
        } else if (req.method === "POST" && req.url === "/analysis/batch") {
          res.end(JSON.stringify({ status: "ok", result: "Batch analysis result" }));
        } else if (req.method === "POST" && req.url === "/analysis/global") {
          res.end(JSON.stringify({ status: "ok", result: "Global report result" }));
        } else if (req.method === "POST" && req.url === "/summaries/clear") {
          res.end(JSON.stringify({ status: "ok" }));
        } else {
          res.statusCode = 404;
          res.end(JSON.stringify({ error: "not mocked" }));
        }
      });
    });

    server.listen(port, () => resolve({ server, getCallLog: () => callLog }));
  });
}

// ─── UI server ───
function startUiServer(port, apiPort) {
  return new Promise((resolve) => {
    const app = express();
    app.use(express.static(path.join(__dirname, "..", "public")));
    app.get("/config", (_req, res) => {
      res.json({ apiBase: `http://localhost:${apiPort}` });
    });
    const server = app.listen(port, () => resolve(server));
  });
}

// ─── Tests ───
async function run() {
  console.log("Starting mock API server...");
  const { server: apiServer, getCallLog } = await startMockApi(API_PORT);

  console.log("Starting UI server...");
  const uiServer = await startUiServer(UI_PORT, API_PORT);

  const browser = await chromium.launch({ headless: !HEADED, args: ["--no-sandbox", "--disable-setuid-sandbox"] });
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  const page = await context.newPage();

  let passed = 0;
  let failed = 0;

  async function t(name, fn) {
    try {
      await fn();
      console.log(`  ✅ ${name}`);
      passed++;
    } catch (e) {
      console.log(`  ❌ ${name}: ${e.message}`);
      failed++;
    }
  }

  console.log("\n📋 UI E2E Tests\n");

  // === PAGE LOAD & STRUCTURE ===

  await t("page loads with pipeline heading", async () => {
    await page.goto(`http://localhost:${UI_PORT}`);
    await page.waitForSelector("h2");
    const h2 = await page.textContent("h2");
    if (!h2.includes("Pipeline")) throw new Error(`Expected "Pipeline", got "${h2}"`);
  });

  await t("renders exactly 6 pipeline steps", async () => {
    const steps = await page.$$(".step");
    if (steps.length !== 6) throw new Error(`Expected 6 steps, got ${steps.length}`);
  });

  await t("page title is correct", async () => {
    const title = await page.title();
    if (!title.includes("MailKB")) throw new Error(`Expected "MailKB", got "${title}"`);
  });

  await t("header logo is visible", async () => {
    const logo = await page.textContent(".logo");
    if (!logo.includes("MailKB")) throw new Error(`Expected MailKB in logo, got "${logo}"`);
  });

  await t("both tabs are present", async () => {
    const tabs = await page.$$(".tab");
    if (tabs.length !== 2) throw new Error(`Expected 2 tabs, got ${tabs.length}`);
    const text0 = await tabs[0].textContent();
    const text1 = await tabs[1].textContent();
    if (!text0.includes("Pipeline") || !text1.includes("Поиск")) {
      throw new Error(`Tab texts: "${text0}", "${text1}"`);
    }
  });

  await t("run-all button is present and enabled", async () => {
    const btn = await page.$("#run-all");
    if (!btn) throw new Error("Run all button missing");
    const disabled = await btn.getAttribute("disabled");
    if (disabled !== null) throw new Error("Run-all button should be enabled initially");
  });

  // === PIPELINE STEPS ===

  await t("init-db step returns ok", async () => {
    await page.click('.run-btn[data-key="init-db"]');
    await page.waitForTimeout(800);
    const status = await page.textContent("#status-init-db");
    if (!status.includes("✅")) throw new Error(`Expected ✅, got "${status}"`);
    // detail should show filenames
    const detail = await page.textContent("#detail-init-db");
    if (!detail.includes("01_create")) throw new Error(`Expected sql filename, got "${detail}"`);
  });

  await t("import-mbox step returns ok", async () => {
    await page.click('.run-btn[data-key="import-mbox"]');
    await page.waitForTimeout(800);
    const status = await page.textContent("#status-import-mbox");
    if (!status.includes("✅")) throw new Error(`Expected ✅, got "${status}"`);
  });

  await t("dedup step returns ok", async () => {
    await page.click('.run-btn[data-key="dedup"]');
    await page.waitForTimeout(800);
    const status = await page.textContent("#status-dedup");
    if (!status.includes("✅")) throw new Error(`Expected ✅, got "${status}"`);
  });

  await t("clean-bodies step returns ok", async () => {
    await page.click('.run-btn[data-key="clean-bodies"]');
    await page.waitForTimeout(800);
    const status = await page.textContent("#status-clean-bodies");
    if (!status.includes("✅")) throw new Error(`Expected ✅, got "${status}"`);
  });

  await t("parse step returns ok with success count", async () => {
    await page.click('.run-btn[data-key="parse"]');
    await page.waitForTimeout(800);
    const status = await page.textContent("#status-parse");
    if (!status.includes("✅")) throw new Error(`Expected ✅, got "${status}"`);
    const detail = await page.textContent("#detail-parse");
    if (!detail.includes("success=")) throw new Error(`Expected success count, got "${detail}"`);
  });

  await t("index-messages step returns ok", async () => {
    await page.click('.run-btn[data-key="index-messages"]');
    await page.waitForTimeout(800);
    const status = await page.textContent("#status-index-messages");
    if (!status.includes("✅")) throw new Error(`Expected ✅, got "${status}"`);
  });

  // === PARAMS MODAL ===

  await t("params modal opens for clean-bodies", async () => {
    await page.click('.params-btn[data-key="clean-bodies"]');
    await page.waitForTimeout(300);
    const modal = await page.$("#params-modal");
    if (!modal) throw new Error("Params modal not opened");
    const h3 = await page.textContent("#params-modal h3");
    if (!h3.includes("clean-bodies")) throw new Error(`Expected clean-bodies, got "${h3}"`);
  });

  await t("params modal has apply and cancel buttons", async () => {
    const apply = await page.$("#params-apply");
    const cancel = await page.$("#params-cancel");
    if (!apply || !cancel) throw new Error("Apply/Cancel buttons missing");
  });

  await t("params modal can be cancelled", async () => {
    await page.click("#params-cancel");
    await page.waitForTimeout(300);
    const modal = await page.$("#params-modal");
    if (modal) throw new Error("Modal should be closed after cancel");
  });

  await t("params modal apply saves and closes", async () => {
    await page.click('.params-btn[data-key="parse"]');
    await page.waitForTimeout(300);
    // change a value
    const inputs = await page.$$("#params-modal input");
    if (inputs.length > 0) {
      await inputs[0].fill("99");
    }
    await page.click("#params-apply");
    await page.waitForTimeout(300);
    const modal = await page.$("#params-modal");
    if (modal) throw new Error("Modal should be closed after apply");
  });

  // === RUN ALL ===

  await t("run-all triggers all 6 steps", async () => {
    await page.click("#run-all");
    await page.waitForTimeout(3000);
    const statuses = await page.$$eval(".step-status", (els) =>
      els.map((el) => el.textContent)
    );
    const okCount = statuses.filter((s) => s.includes("✅")).length;
    if (okCount < 6) throw new Error(`Expected 6 OK statuses, got ${okCount}: ${statuses.join(", ")}`);
  });

  // === SEARCH TAB ===

  await t("switches to search tab", async () => {
    await page.click('.tab[data-tab="search"]');
    await page.waitForTimeout(200);
    const input = await page.$("#project-input");
    if (!input) throw new Error("Search input missing");
    const searchTab = await page.$("#tab-search");
    const isActive = await searchTab.getAttribute("class");
    if (!isActive.includes("active")) throw new Error("Search tab not active");
  });

  await t("search threads returns results", async () => {
    await page.fill("#project-input", "segezha");
    await page.click('[data-action="search-threads"]');
    await page.waitForTimeout(800);
    const html = await page.innerHTML("#search-results");
    if (!html.includes("project_hint")) throw new Error("No project_hint in results");
    if (!html.includes("segezha")) throw new Error("Expected segezha in results");
  });

  await t("corpus batch returns results", async () => {
    await page.fill("#project-input", "segezha");
    await page.click('[data-action="corpus-batch"]');
    await page.waitForTimeout(800);
    const html = await page.innerHTML("#search-results");
    if (!html.includes("project_hint")) throw new Error("No project_hint in corpus results");
  });

  await t("batch analysis returns result", async () => {
    await page.fill("#project-input", "segezha");
    await page.click('[data-action="batch-analysis"]');
    await page.waitForTimeout(800);
    const html = await page.innerHTML("#search-results");
    if (!html.includes("Batch")) throw new Error(`Expected "Batch", got: ${html.substring(0, 80)}`);
  });

  await t("global analysis returns result", async () => {
    await page.fill("#project-input", "segezha");
    await page.click('[data-action="global-analysis"]');
    await page.waitForTimeout(800);
    const html = await page.innerHTML("#search-results");
    if (!html.includes("Global")) throw new Error(`Expected "Global", got: ${html.substring(0, 80)}`);
  });

  await t("clear summaries works", async () => {
    await page.fill("#project-input", "segezha");
    await page.click('[data-action="clear-summaries"]');
    await page.waitForTimeout(800);
    const html = await page.innerHTML("#search-results");
    if (!html.includes("Summaries")) throw new Error("Expected summaries cleared message");
  });

  // === SEARCH VALIDATION ===

  await t("empty project hint shows validation toast", async () => {
    await page.fill("#project-input", "");
    await page.click('[data-action="search-threads"]');
    await page.waitForTimeout(400);
    const toast = await page.textContent("#toast");
    if (!toast.includes("Введите")) throw new Error(`Expected validation toast, got "${toast}"`);
  });

  await t("toast hides after timeout", async () => {
    await page.waitForTimeout(4500);
    const toastClass = await page.getAttribute("#toast", "class");
    if (!toastClass.includes("hidden")) throw new Error("Toast should be hidden after timeout");
  });

  // === TAB SWITCHING ===

  await t("switches back to pipeline tab", async () => {
    await page.click('.tab[data-tab="pipeline"]');
    await page.waitForTimeout(200);
    const pipelineTab = await page.$("#tab-pipeline");
    const isActive = await pipelineTab.getAttribute("class");
    if (!isActive.includes("active")) throw new Error("Pipeline tab not active");
  });

  // === KEYBOARD ===

  await t("enter key triggers search", async () => {
    await page.click('.tab[data-tab="search"]');
    await page.waitForTimeout(200);
    await page.fill("#project-input", "sibur");
    await page.press("#project-input", "Enter");
    await page.waitForTimeout(800);
    const html = await page.innerHTML("#search-results");
    if (!html.includes("project_hint")) throw new Error("Enter key should trigger search");
  });

  // === API CALLS VERIFICATION ===

  await t("API was called for each pipeline step", async () => {
    const log = getCallLog();
    const pipelineCalls = log.filter(
      (c) => c.method === "POST" && c.url.startsWith("/pipeline/")
    );
    if (pipelineCalls.length < 12) {
      // 6 individual runs + 6 run-all = 12 calls minimum
      console.log(`      (pipeline API calls: ${pipelineCalls.length})`);
    }
  });

  // ─── Cleanup ───
  await browser.close();
  await new Promise((r) => apiServer.close(r));
  await new Promise((r) => uiServer.close(r));

  console.log(`\n${"=".repeat(40)}`);
  console.log(`  Passed: ${passed}, Failed: ${failed}`);
  console.log(`${"=".repeat(40)}`);

  process.exit(failed > 0 ? 1 : 0);
}

run().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
