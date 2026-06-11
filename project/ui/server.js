const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
const API_BASE = process.env.API_BASE || "http://localhost:8010";

app.use(express.static(path.join(__dirname, "public")));

app.get("/config", (_req, res) => {
  res.json({ apiBase: API_BASE });
});

app.listen(PORT, () => {
  console.log(`MailKB UI server running on http://0.0.0.0:${PORT}`);
  console.log(`API backend: ${API_BASE}`);
});
