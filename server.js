// server.js
require("dotenv").config();
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const vision = require("@google-cloud/vision");

const app = express();
const PORT = process.env.PORT || 3000;

// ────────────────────────────────────────────────────────────────────────────────
// 1) GOOGLE VISION SETUP & CREDENTIAL PARSING
// ────────────────────────────────────────────────────────────────────────────────
if (!process.env.GOOGLE_CLOUD_CREDENTIALS) {
  console.error("❌ GOOGLE_CLOUD_CREDENTIALS env var not set");
  process.exit(1);
}

let credentials;
try {
  credentials = JSON.parse(process.env.GOOGLE_CLOUD_CREDENTIALS);
} catch (err) {
  console.error("❌ Invalid GOOGLE_CLOUD_CREDENTIALS JSON:", err.message);
  process.exit(1);
}

const client = new vision.ImageAnnotatorClient({ credentials });

// ────────────────────────────────────────────────────────────────────────────────
// 2) MULTER CONFIGURATION (accepts field "receipt")
// ────────────────────────────────────────────────────────────────────────────────
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadDir),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname);
    const base = path.basename(file.originalname, ext);
    cb(null, `${Date.now()}-${base}${ext}`);
  },
});

const fileFilter = (req, file, cb) => {
  console.log("📦 Incoming file:", { originalname: file.originalname, mimetype: file.mimetype });
  if (file.mimetype.startsWith("image/") || file.mimetype === "application/octet-stream") {
    return cb(null, true);
  }
  return cb(new Error("Only image files allowed"), false);
};

const upload = multer({
  storage,
  fileFilter,
});

// ────────────────────────────────────────────────────────────────────────────────
// 3) LOGGING UTILITIES (structured)
// ────────────────────────────────────────────────────────────────────────────────
const logsDir = path.join(__dirname, "logs");
if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });
const errorLogFile = path.join(logsDir, "ocr-errors.log");

function logError(message, content) {
  const timestamp = new Date().toISOString();
  const entry = { timestamp, level: "error", message, content };
  fs.appendFileSync(errorLogFile, JSON.stringify(entry) + "\n");
}

// ────────────────────────────────────────────────────────────────────────────────
// 4) HELPER: PROMISE WITH TIMEOUT & OCR WITH RETRIES
// ────────────────────────────────────────────────────────────────────────────────
function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error("OCR timeout")), ms))
  ]);
}

async function performOcrWithRetry(filePath, attempts = 2, timeoutMs = 10000) {
  let lastErr;
  for (let i = 1; i <= attempts; i++) {
    try {
      const [result] = await withTimeout(
        client.documentTextDetection(filePath),
        timeoutMs
      );
      return result;
    } catch (err) {
      lastErr = err;
      logError(`OCR attempt ${i} failed`, err.stack || err.message);
      if (i === attempts) throw lastErr;
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────────
// 5) ROUTE: POST /parse-receipt
// ────────────────────────────────────────────────────────────────────────────────
app.post("/parse-receipt", upload.single("receipt"), async (req, res) => {
  const file = req.file;
  if (!file) {
    return res.status(400).json({ status: "error", message: "No file uploaded under field 'receipt'." });
  }
  const imagePath = file.path;

  try {
    const docResult = await performOcrWithRetry(imagePath);
    const fullText =
      docResult?.fullTextAnnotation?.text ||
      docResult?.textAnnotations?.[0]?.description ||
      "";

    if (!fullText.trim()) {
      logError("⚠️ OCR returned empty text", `File: ${file.filename}`);
      try { fs.unlinkSync(imagePath); } catch (delErr) { logError("File delete error", delErr.message); }
      return res.json({ status: "ok", data: { merchant: null, date: null, total: null } });
    }

    const lines = fullText.split("\n").map(l => l.trim()).filter(Boolean);
    const { merchant, merchantConf } = extractMerchant(lines);
    const { date, dateConf } = extractDate(lines);
    const { total, totalConf } = extractTotal(lines);

    try { fs.unlinkSync(imagePath); } catch (delErr) { logError("File delete error", delErr.message); }

    return res.json({
      status: "ok",
      data: {
        merchant: { value: merchant, confidence: merchantConf },
        date:     { value: date,     confidence: dateConf     },
        total:    { value: total,    confidence: totalConf    }
      }
    });
  } catch (err) {
    const code = err.message === "OCR timeout" ? 504 : 500;
    logError("🚨 OCR failure or timeout", err.stack || err.message);
    try { fs.unlinkSync(imagePath); } catch (delErr) { logError("File delete error", delErr.message); }
    return res.status(code).json({ status: "error", message: err.message || "Server error" });
  }
});

// ────────────────────────────────────────────────────────────────────────────────
// 6) HEALTH & READINESS ENDPOINTS
// ────────────────────────────────────────────────────────────────────────────────
app.get("/healthz", (_req, res) => {
  res.status(200).json({ status: "ok" });
});
app.get("/readyz", async (_req, res) => {
  try {
    await client.getProjectId();
    res.status(200).json({ status: "ok" });
  } catch (err) {
    res.status(503).json({ status: "error", message: "Vision client not ready" });
  }
});

// ────────────────────────────────────────────────────────────────────────────────
// 7) GRACEFUL SHUTDOWN
// ────────────────────────────────────────────────────────────────────────────────
const server = app.listen(PORT, () => console.log(`🧾 Server listening on http://localhost:${PORT}`));

const shutdown = () => {
  console.log("⚠️  Shutdown signal received, closing... ");
  server.close(() => {
    console.log("✅ Closed out remaining connections");
    process.exit(0);
  });
  setTimeout(() => {
    console.error("⚠️  Could not close connections in time, forcing exit");
    process.exit(1);
  }, 30000);
};
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);

// ────────────────────────────────────────────────────────────────────────────────
// HELPER FUNCTIONS: extractMerchant, extractDate, extractTotal
// ────────────────────────────────────────────────────────────────────────────────

/* ... existing extractMerchant, extractDate, extractTotal ... */


/**
 * extractMerchant(lines) → { merchant: string|null, merchantConf: number }
 *
 *  • If we find a line that is ALL CAPS (no digits) and not “FACTURA/IVA/TICKET/TOTAL” → conf 0.90
 *  • Else if we detect any known chain from the expanded list → conf 0.90
 *  • Else fallback to lines[0] → conf 0.50
 *  • If lines is empty → merchant=null, conf=0.00
 */
function extractMerchant(lines) {
  if (!lines.length) return { merchant: null, merchantConf: 0.0 };

  // Helper: “almost entirely uppercase, no digits”
  const isAllCapsNoDigits = (line) => {
    if (!/[A-ZÁÉÍÓÚÑ]/.test(line)) return false;
    if (/\d/.test(line)) return false;
    const chars = line.replace(/\s+/g, "");
    let countCaps = 0;
    for (let ch of chars) {
      if (/[A-ZÁÉÍÓÚÑ,.\-&]/.test(ch)) countCaps++;
    }
    return countCaps / Math.max(chars.length, 1) > 0.8;
  };

  // Skip lines that are clearly not a merchant
  const skipPattern = /FACTURA|TICKET|IVA|TOTAL/i;
  for (let line of lines) {
    if (isAllCapsNoDigits(line) && !skipPattern.test(line)) {
      return { merchant: line, merchantConf: 0.9 };
    }
  }

  // Expanded list of known Spanish chains:
  const knownChains = [
    "MERCADONA",
    "CARREFOUR",
    "LIDL",
    "ALDI",
    "ALCAMPO",
    "EROSKI",
    "CONSUM",
    "HIPERCOR",
    "SPAR",
    "COVIRÁN",
    "SUPERSOL",
    "AHORRAMAS",
    "FROIZ",
    "GADIS",
    "CAPRABO",
    "BON PREU",
    "LA UNION DE TARRAGONA",
    "SUPERDINO",
    "EL CORTE INGLÉS",
    "CARREFOUR EXPRESS",
    "SUPECAMPO",
    "SUPERDT",
    "CASA IBAÑEZ",
    "DÍA",
    "BM SUPERMERCADOS",
    "FAMILY CASH",
    "SUPERCONSUM",
    "HIPERUSERA",
    "MARKET 7",
    "SUPER 7",
    "LA DESPENSA",
    "LIDL SUPERMERCADOS",
    "CONSUM SUPERMERCADOS",
    "CARREFOUR MARKET",
    "SUPER USERA",
    "ALIMERKA",
    "E.LECLERC",
    "SUPERCOR",
    "HIPERCO",
    "MERCADONA S.A.",
    "HIPERDINO",
  ];

  for (let line of lines) {
    for (let chain of knownChains) {
      if (line.toUpperCase().includes(chain)) {
        return { merchant: line, merchantConf: 0.9 };
      }
    }
  }

  // Fallback: first non-empty line
  return { merchant: lines[0], merchantConf: 0.5 };
}

/**
 * extractDate(lines) → { date: string|null, dateConf: number }
 *
 *  • Matches DD/MM/YYYY, DD/MM YYYY, DD.MM.YYYY, DD-MM-YYYY, etc. → conf 0.90
 *  • Else → date=null, conf=0.00
 */
function extractDate(lines) {
  const dateRegex = /\b(\d{1,2})[\/\.\-\s](\d{1,2})(?:[\/\.\-\s])?(\d{2,4})\b/;

  for (let line of lines) {
    const m = line.match(dateRegex);
    if (!m) continue;

    let [_, d, mth, y] = m;

    // Normalize
    if (y.length === 2) y = "20" + y;
    if (d.length === 1) d = "0" + d;
    if (mth.length === 1) mth = "0" + mth;

    const day = parseInt(d, 10);
    const month = parseInt(mth, 10);
    const year = parseInt(y, 10);

    // Validate ranges
    const isValidDay = day >= 1 && day <= 31;
    const isValidMonth = month >= 1 && month <= 12;
    const isValidYear = year >= 2000 && year <= 2100;

    if (isValidDay && isValidMonth && isValidYear) {
      return { date: `${d}/${mth}/${y}`, dateConf: 0.9 };
    } else {
      logError(
        "⛔ Rejected date",
        `Line: "${line}"\nParsed: ${d}/${mth}/${y} — Invalid range.\n`
      );
    }
  }

  return { date: null, dateConf: 0.0 };
}


/**
 * extractTotal(lines) → { total: string|null, totalConf: number }
 *
 *  1) “TOTAL … (€) amount” inline → conf 0.90
 *  2) “IMPORTE … amount” inline → conf 0.80
 *  3) FALLBACK: match any decimal (e.g. “1.1”, “2,2”, “20.0”, “16.50”), pick largest → conf 0.60
 *  4) If none found → total=null, conf=0.00
 */
function extractTotal(lines) {
  // 1) TOTAL (€) or TOTAL … amount on same line
  const totalInlineRegex = /\bTOTAL(?:\s*\(€\))?[^0-9]*?(\d{1,3}[.,]\d{2})/i;
  for (let line of lines) {
    const m = line.match(totalInlineRegex);
    if (m) {
      const out = m[1].includes(",") ? m[1] : m[1].replace(".", ",");
      return { total: out, totalConf: 0.9 };
    }
  }

  // 2) IMPORTE A ABONAR or IMPORTE … amount
  const importeRegex = /\bIMPORTE(?:\s+A\s+ABONAR)?[^0-9]*?(\d{1,3}[.,]\d{2})/i;
  for (let line of lines) {
    const m = line.match(importeRegex);
    if (m) {
      const out = m[1].includes(",") ? m[1] : m[1].replace(".", ",");
      return { total: out, totalConf: 0.8 };
    }
  }

  // 3) Fallback → any decimal (one or more digits + dot/comma + one or more digits)
  const anyDecimalRegex = /(\d+(?:[.,]\d+))/g;
  let allAmounts = [];
  for (let line of lines) {
    let mm;
    while ((mm = anyDecimalRegex.exec(line)) !== null) {
      const candidate = mm[1];
      if (/^\d+[.,]\d+$/.test(candidate)) {
        const num = parseFloat(candidate.replace(",", "."));
        if (!isNaN(num)) {
          allAmounts.push(num);
        }
      }
    }
  }
  if (allAmounts.length) {
    const maxVal = Math.max(...allAmounts);
    const formatted = maxVal.toFixed(2).replace(".", ",");
    return { total: formatted, totalConf: 0.6 };
  }

  return { total: null, totalConf: 0.0 };
}
