import express from "express";
import cors from "cors";
import { randomUUID } from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_FILE = path.join(__dirname, "data.json");
const CATEGORIES = ["כוח", "אומנויות לחימה", "סלסה", "גיטרה", "צילום/כלב/SUP"];
const MODEL = "gemini-3.8-flash"; // free tier via Google AI Studio (ai.google.dev)

const app = express();
app.set("trust proxy", 1); // Render/Railway sit behind a proxy; needed for req.ip to be accurate
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

function loadData() {
  try { return JSON.parse(fs.readFileSync(DATA_FILE, "utf8")); } catch { return {}; }
}
function saveData(data) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
}

function userId(req) {
  const id = req.header("x-user-id");
  if (!id) throw Object.assign(new Error("missing x-user-id"), { status: 400 });
  return id;
}

// --- Rate limiting -------------------------------------------------------
const WINDOW_MS = 10 * 60 * 1000;
const MAX_REQUESTS_PER_WINDOW = 15;
const hits = new Map();
function rateLimit(req, res, next) {
  const ip = req.ip;
  const now = Date.now();
  const recent = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  if (recent.length >= MAX_REQUESTS_PER_WINDOW) {
    return res.status(429).json({ error: "יותר מדי בקשות, נסה שוב בעוד כמה דקות." });
  }
  recent.push(now);
  hits.set(ip, recent);
  next();
}
const MAX_TEXT_LENGTH = 500;

// --- Gemini API helper ----------------------------------------------------
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;

// The free tier gets deprioritized under load and returns 503 "high demand"
// fairly often. Retry a couple of times with a short backoff before giving
// up — this is standard practice for any app calling a third-party LLM API.
async function fetchWithRetry(url, options, retries = 2) {
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(url, options);
    if (res.status !== 503 || attempt >= retries) return res;
    await new Promise((r) => setTimeout(r, 800 * (attempt + 1)));
  }
}

async function callGemini(text, { json = false } = {}) {
  const res = await fetchWithRetry(GEMINI_URL, {
    method: "POST",
    headers: { "content-type": "application/json", "x-goog-api-key": process.env.GEMINI_API_KEY },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text }] }],
      ...(json ? { generationConfig: { responseMimeType: "application/json" } } : {}),
    }),
  });
  if (!res.ok) throw new Error(`Gemini API error: ${res.status} ${await res.text()}`);
  const data = await res.json();
  const out = data.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("\n") || "";
  return json ? JSON.parse(out) : out;
}

// --- Agent: real tool use (function calling) ------------------------------
// search_journal is keyword-based retrieval (RAG-style: retrieve, then
// generate) rather than true vector/embedding search — real semantic search
// needs an embeddings model + vector store, a natural next upgrade.
const TOOLS = [{
  functionDeclarations: [
    {
      name: "search_journal",
      description: "מחפש רשומות רלוונטיות ביומן האישי של המשתמש לפי מילות מפתח מהשאלה.",
      parameters: {
        type: "OBJECT",
        properties: { query: { type: "STRING", description: "מילות החיפוש" } },
        required: ["query"],
      },
    },
    {
      name: "get_weather",
      description: "מחזיר תחזית רוח, משקעים וגובה גלים (ים) לאזור תל אביב, שימושי להמלצות על גלישת SUP.",
      parameters: { type: "OBJECT", properties: {} },
    },
  ],
}];

function searchJournal(entries, query) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const scored = entries.map((e) => {
    const hay = `${e.text} ${e.summary}`.toLowerCase();
    const score = words.reduce((s, w) => s + (hay.includes(w) ? 1 : 0), 0);
    return { e, score };
  });
  return scored.filter((s) => s.score > 0).sort((a, b) => b.score - a.score).slice(0, 5).map((s) => s.e);
}

async function getWeather() {
  const result = {};
  try {
    const r = await fetch("https://api.open-meteo.com/v1/forecast?latitude=32.08&longitude=34.78&current=wind_speed_10m,precipitation&daily=wind_speed_10m_max&timezone=Asia%2FJerusalem");
    const d = await r.json();
    result.wind_now_kmh = d.current?.wind_speed_10m;
    result.precipitation_now_mm = d.current?.precipitation;
    result.wind_tomorrow_max_kmh = d.daily?.wind_speed_10m_max?.[1];
  } catch {
    result.wind_error = "לא הצלחתי לקבל נתוני רוח כרגע.";
  }
  // Wave height matters more than wind for SUP/surf conditions — separate
  // free API (Open-Meteo Marine), so it's fetched and merged independently.
  try {
    const r = await fetch("https://marine-api.open-meteo.com/v1/marine?latitude=32.08&longitude=34.78&current=wave_height&daily=wave_height_max&timezone=Asia%2FJerusalem");
    const d = await r.json();
    result.wave_height_now_m = d.current?.wave_height;
    result.wave_height_tomorrow_max_m = d.daily?.wave_height_max?.[1];
  } catch {
    result.wave_error = "לא הצלחתי לקבל נתוני גלים כרגע.";
  }
  return result;
}

// Tool-use loop: sends the conversation + tool definitions, executes any
// function the model asks for, feeds the result back, repeats (capped)
// until the model gives a final text answer.
async function runAgent(userText, entries) {
  let contents = [{ role: "user", parts: [{ text: userText }] }];
  for (let i = 0; i < 4; i++) {
    const res = await fetchWithRetry(GEMINI_URL, {
      method: "POST",
      headers: { "content-type": "application/json", "x-goog-api-key": process.env.GEMINI_API_KEY },
      body: JSON.stringify({ contents, tools: TOOLS }),
    });
    if (!res.ok) throw new Error(`Gemini API error: ${res.status} ${await res.text()}`);
    const data = await res.json();
    const parts = data.candidates?.[0]?.content?.parts || [];
    const calls = parts.filter((p) => p.functionCall);

    if (!calls.length) {
      return parts.map((p) => p.text || "").join("\n") || "לא הצלחתי לנסח תשובה.";
    }

    contents.push({ role: "model", parts });
    const responseParts = [];
    for (const p of calls) {
      const { name, args } = p.functionCall;
      let result;
      if (name === "search_journal") result = searchJournal(entries, args?.query || "");
      else if (name === "get_weather") result = await getWeather();
      else result = { error: "unknown tool" };
      responseParts.push({ functionResponse: { name, response: { result } } });
    }
    contents.push({ role: "user", parts: responseParts });
  }
  return "לא הצלחתי לסיים את המשימה, נסה לנסח מחדש.";
}

// --- Routes ----------------------------------------------------------------
app.get("/api/entries", (req, res) => {
  const data = loadData();
  res.json(data[userId(req)]?.entries || []);
});

app.post("/api/entries", rateLimit, async (req, res) => {
  try {
    const id = userId(req);
    const text = (req.body.text || "").trim().slice(0, MAX_TEXT_LENGTH);
    if (!text) return res.status(400).json({ error: "empty text" });

    let parsed = { categories: [], summary: text };
    try {
      parsed = await callGemini(
        `סווג את הטקסט הבא לקטגוריה אחת או יותר מתוך: ${CATEGORIES.join(", ")}. ` +
        `החזר JSON בלבד: {"categories": ["..."], "summary": "משפט קצר בעברית"}. הטקסט: """${text}"""`,
        { json: true }
      );
    } catch (e) { console.error("classify failed", e.message); }

    const entry = {
      id: randomUUID(),
      text,
      categories: parsed.categories || [],
      summary: parsed.summary || text,
      createdAt: Date.now(),
    };

    const data = loadData();
    if (!data[id]) data[id] = { entries: [] };
    data[id].entries.unshift(entry);
    data[id].entries = data[id].entries.slice(0, 200);
    saveData(data);

    res.json(entry);
  } catch (e) {
    res.status(e.status || 500).json({ error: e.message });
  }
});

app.post("/api/summary", rateLimit, async (req, res) => {
  try {
    const id = userId(req);
    const data = loadData();
    const entries = data[id]?.entries || [];
    if (!entries.length) return res.json({ summary: "אין עדיין מספיק רשומות לסיכום." });

    const recent = entries.slice(0, 30).map((e) => `- ${e.summary}`).join("\n");
    const summary = await callGemini(
      `אתה מאמן אישי חכם וחברותי. בהתבסס על הפעילויות האחרונות, כתוב סיכום קצר (עד 6 שורות) בעברית: ` +
      `מה בלט, איזה תחום הוזנח, והצעה אחת קונקרטית לשבוע הקרוב:\n${recent}`
    );
    res.json({ summary });
  } catch (e) {
    res.status(e.status || 500).json({ error: e.message });
  }
});

app.post("/api/ask", rateLimit, async (req, res) => {
  try {
    const id = userId(req);
    const question = (req.body.question || "").trim().slice(0, MAX_TEXT_LENGTH);
    if (!question) return res.status(400).json({ error: "empty question" });
    const data = loadData();
    const entries = data[id]?.entries || [];
    const answer = await runAgent(question, entries);
    res.json({ answer });
  } catch (e) {
    res.status(e.status || 500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on ${PORT}`));
