"use strict";

/* AI Agents Console: plain JavaScript, no build step. Every call goes to the same FastAPI server:
 *   /health, /agents                 status
 *   /agents/{search,data}/stream_events  chat with live tool steps (LangServe, Server-Sent Events)
 *   /agents/summarize/invoke         summarize chain
 *   /items                           CRUD (SQL Server)
 *   /passengers, /stats/*            Titanic data (SQL Server)
 */

// ------------------------------------------------------------------ helpers

const $ = (selector, root = document) => root.querySelector(selector);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== undefined && value !== null && value !== false) node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function errorMessage(status, body) {
  const detail = body && body.detail;
  if (Array.isArray(detail)) {
    // Pydantic validation errors: [{loc: ["body", "price"], msg: "..."}]
    return detail.map((d) => `${(d.loc || []).slice(1).join(".") || "input"}: ${d.msg}`).join("; ");
  }
  if (typeof detail === "string") return detail;
  return `Request failed (HTTP ${status})`;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (res.status === 204) return null;
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorMessage(res.status, body));
  return body;
}

const fmt = (value) =>
  typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value;

/** columns: [key, label, {num: bool, render: (row) => Node|string}] */
function renderTable(rows, columns) {
  if (!rows.length) return el("p", { class: "hint" }, "No rows.");
  const head = el("tr", {}, columns.map(([, label, opt = {}]) => el("th", { class: opt.num ? "num" : "" }, label)));
  const body = rows.map((row) =>
    el("tr", {}, columns.map(([key, , opt = {}]) =>
      el("td", { class: opt.num ? "num" : "" }, opt.render ? opt.render(row) : fmt(row[key] ?? "")),
    )),
  );
  return el("table", { class: "data" }, el("thead", {}, head), el("tbody", {}, body));
}

// ------------------------------------------------------------------ safe markdown for answers
// Everything is escaped first; only a few patterns (bold, code, links, lists, tables) become tags.

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function inline(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  return html;
}

function renderMarkdown(text) {
  const lines = String(text).replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*\|.*\|\s*$/.test(line)) {
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
      const cells = (row) => row.trim().slice(1, -1).split("|").map((c) => c.trim());
      const body = rows.filter((row) => !/^\s*\|[\s:|-]+\|\s*$/.test(row));
      const [header, ...rest] = body;
      out.push(
        "<table><thead><tr>" + cells(header).map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
        rest.map((r) => "<tr>" + cells(r).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>",
      );
    } else if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
        items.push(lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, ""));
      }
      out.push("<ul>" + items.map((item) => `<li>${inline(item)}</li>`).join("") + "</ul>");
    } else if (line.trim() === "") {
      i++;
    } else {
      const para = [];
      while (i < lines.length && lines[i].trim() !== "" && !/^\s*(\||[-*]\s|\d+\.\s)/.test(lines[i])) {
        para.push(lines[i++]);
      }
      if (!para.length) para.push(lines[i++]);
      out.push(`<p>${para.map(inline).join("<br>")}</p>`);
    }
  }
  return out.join("");
}

// ------------------------------------------------------------------ tabs + status

const loaders = {};

function showTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((panel) => { panel.hidden = panel.id !== `tab-${name}`; });
  if (loaders[name]) loaders[name]();
  try { localStorage.setItem("tab", name); } catch { /* storage may be blocked */ }
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));

let agentsEnabled = false;

async function loadStatus() {
  const dbPill = $("#status-db");
  const agentsPill = $("#status-agents");
  try {
    const health = await api("/health");
    const ok = health.storage === "memory" || health.database === "ok";
    dbPill.textContent = health.storage === "memory" ? "Storage: memory" : `Database: ${health.database}`;
    dbPill.className = `pill ${ok ? "ok" : "bad"}`;
  } catch {
    dbPill.textContent = "Server unreachable";
    dbPill.className = "pill bad";
  }
  try {
    const agents = await api("/agents");
    agentsEnabled = Object.values(agents).every((a) => a.invoke !== "disabled");
    agentsPill.textContent = agentsEnabled ? "Agents: ready" : "Agents: off (no GROQ_API_KEY)";
    agentsPill.className = `pill ${agentsEnabled ? "ok" : "bad"}`;
  } catch {
    agentsPill.textContent = "Agents: unknown";
    agentsPill.className = "pill bad";
  }
  renderChat();
}

// ------------------------------------------------------------------ chat

const AGENTS = {
  search: {
    hint: "Searches Google through SerpAPI (at most 3 searches per question) and remembers this conversation.",
    examples: ["What is the latest version of Python?", "Who is the CEO of Groq?", "What is LangSmith Studio?"],
  },
  data: {
    hint: "Answers from the Titanic tables in SQL Server using the 5 Week 1 analytics queries.",
    examples: ["Which class and sex had the best survival rate?", "Did travelling alone affect survival?", "Who were the oldest passengers?"],
  },
};

const chat = {
  agent: "search",
  threads: { search: null, data: null },   // thread_id from the server = the agent's memory
  messages: { search: [], data: [] },      // what is shown in the log
  busy: false,
};

const chatLog = $("#chat-log");

function currentAgent() {
  return document.querySelector('input[name="agent"]:checked').value;
}

function renderMessage(msg) {
  if (msg.role === "user") return el("div", { class: "msg user" }, msg.text);
  if (msg.role === "error") return el("div", { class: "msg error" }, msg.text);
  const nodes = [];
  if (msg.steps && msg.steps.length) {
    nodes.push(el("div", { class: "steps" }, msg.steps.map((step) =>
      el("div", { class: "step" }, `🔧 ${step.label}`, step.done ? el("span", { class: "done" }, "  ✓") : " …"),
    )));
  }
  const bubble = el("div", { class: `msg agent${msg.pending ? " typing" : ""}` });
  if (msg.pending) bubble.textContent = msg.text || (msg.steps?.length ? "" : "Thinking");
  else bubble.innerHTML = renderMarkdown(msg.text);
  if (msg.text || msg.pending) nodes.push(bubble);
  return nodes;
}

function renderChat() {
  const agent = chat.agent;
  $("#agent-hint").textContent = AGENTS[agent].hint;
  chatLog.replaceChildren();
  const messages = chat.messages[agent];
  if (!messages.length) {
    const empty = el("div", { class: "empty" },
      agentsEnabled ? el("p", {}, "Try one of these:") : el("p", {}, "Agents are off: add GROQ_API_KEY to .env and restart the server."),
      agentsEnabled ? el("ul", {}, AGENTS[agent].examples.map((q) =>
        el("li", {}, el("button", { class: "btn small ghost", type: "button", onclick: () => ask(q) }, q)))) : null,
    );
    chatLog.append(empty);
  }
  for (const msg of messages) chatLog.append(...[renderMessage(msg)].flat());
  chatLog.classList.toggle("hide-steps", !$("#show-steps").checked);
  chatLog.scrollTop = chatLog.scrollHeight;
  const thread = chat.threads[agent];
  $("#thread-meta").textContent = thread ? `Conversation id (thread_id): ${thread}` : "New conversation";
  $("#chat-send").disabled = chat.busy || !agentsEnabled;
}

function describeTool(name, input) {
  if (name === "google_search" && input && input.query) return `google_search("${input.query}")`;
  const args = input && Object.keys(input).length ? JSON.stringify(input) : "";
  return `${name}(${args})`;
}

/** POST to LangServe's stream_events and call onEvent for every event. */
async function streamEvents(path, input, onEvent) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input }),
  });
  if (!res.ok) throw new Error(errorMessage(res.status, await res.json().catch(() => null)));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Server-Sent Events: blocks separated by a blank line, each with "event:" and "data:" lines.
    let match;
    while ((match = /\r?\n\r?\n/.exec(buffer))) {
      const block = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      let type = "message";
      let data = "";
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith("event:")) type = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (type === "end") return;
      if (type === "error") throw new Error(errorMessage(500, JSON.parse(data || "{}")));
      if (data) onEvent(JSON.parse(data));
    }
  }
}

async function ask(question) {
  question = question.trim();
  if (!question || chat.busy) return;
  const agent = chat.agent;
  const messages = chat.messages[agent];
  messages.push({ role: "user", text: question });
  const reply = { role: "agent", text: "", steps: [], pending: true };
  messages.push(reply);
  chat.busy = true;
  renderChat();

  const toolSteps = new Map();   // run_id -> step
  try {
    await streamEvents(`/agents/${agent}/stream_events`, { question, thread_id: chat.threads[agent] }, (event) => {
      const { event: kind, data = {}, run_id: runId, parent_ids: parents = [] } = event;
      if (kind === "on_tool_start") {
        const step = { label: describeTool(event.name, data.input), done: false };
        toolSteps.set(runId, step);
        reply.steps.push(step);
      } else if (kind === "on_tool_end" && toolSteps.has(runId)) {
        toolSteps.get(runId).done = true;
      } else if (kind === "on_chat_model_start") {
        reply.text = "";   // a new model turn; earlier turns were tool calls
      } else if (kind === "on_chat_model_stream") {
        const content = data.chunk && data.chunk.content;
        if (typeof content === "string") reply.text += content;
      } else if (kind === "on_chain_end" && parents.length === 0 && data.output) {
        // The top-level run finished: this is the cleaned answer plus the conversation id.
        reply.text = data.output.answer;
        chat.threads[agent] = data.output.thread_id;
      } else {
        return;
      }
      if (chat.agent === agent) renderChat();
    });
  } catch (err) {
    messages.push({ role: "error", text: `Error: ${err.message}` });
  } finally {
    reply.pending = false;
    if (!reply.text && !reply.steps.length) messages.splice(messages.indexOf(reply), 1);
    chat.busy = false;
    if (chat.agent === agent) renderChat();
  }
}

$("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const box = $("#chat-question");
  const question = box.value;
  box.value = "";
  ask(question);
});
$("#chat-question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#chat-form").requestSubmit();
  }
});
document.querySelectorAll('input[name="agent"]').forEach((radio) =>
  radio.addEventListener("change", () => { chat.agent = currentAgent(); renderChat(); }));
$("#new-chat").addEventListener("click", () => {
  chat.threads[chat.agent] = null;
  chat.messages[chat.agent] = [];
  renderChat();
  $("#chat-question").focus();
});
$("#show-steps").addEventListener("change", renderChat);

// ------------------------------------------------------------------ summarize

$("#summarize-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const result = $("#sum-result");
  const button = e.submitter;
  result.hidden = false;
  result.textContent = "Summarizing…";
  button.disabled = true;
  try {
    const body = { input: { text: $("#sum-text").value, max_words: Number($("#sum-words").value) } };
    const data = await api("/agents/summarize/invoke", { method: "POST", body: JSON.stringify(body) });
    result.textContent = data.output;
  } catch (err) {
    result.textContent = `Error: ${err.message}`;
  } finally {
    button.disabled = false;
  }
});

// ------------------------------------------------------------------ items (CRUD)

let editingId = null;

function resetItemForm() {
  editingId = null;
  $("#item-form").reset();
  $("#item-form-title").textContent = "Add an item";
  $("#item-submit").textContent = "Add item";
  $("#item-cancel").hidden = true;
  $("#item-error").textContent = "";
}

function startEdit(item) {
  editingId = item.id;
  $("#item-name").value = item.name;
  $("#item-price").value = item.price;
  $("#item-desc").value = item.description || "";
  $("#item-tags").value = item.tags.join(", ");
  $("#item-form-title").textContent = `Edit item #${item.id}`;
  $("#item-submit").textContent = "Save changes";
  $("#item-cancel").hidden = false;
  $("#item-name").focus();
}

async function deleteItem(item) {
  if (!confirm(`Delete "${item.name}"?`)) return;
  try {
    await api(`/items/${item.id}`, { method: "DELETE" });
    if (editingId === item.id) resetItemForm();
    loadItems();
  } catch (err) {
    alert(err.message);
  }
}

async function loadItems() {
  const wrap = $("#items-table");
  try {
    const items = await api("/items?limit=100");
    wrap.replaceChildren(renderTable(items, [
      ["id", "ID", { num: true }],
      ["name", "Name"],
      ["price", "Price", { num: true, render: (r) => r.price.toFixed(2) }],
      ["tags", "Tags", { render: (r) => el("span", {}, r.tags.map((t) => el("span", { class: "tag" }, t))) }],
      ["description", "Description"],
      ["updated_at", "Updated", { render: (r) => new Date(r.updated_at || r.created_at).toLocaleString() }],
      ["", "", { render: (r) => el("span", {},
        el("button", { class: "btn small", type: "button", onclick: () => startEdit(r) }, "Edit"), " ",
        el("button", { class: "btn small danger", type: "button", onclick: () => deleteItem(r) }, "Delete")) }],
    ]));
  } catch (err) {
    wrap.replaceChildren(el("p", { class: "form-error" }, err.message));
  }
}

$("#item-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    name: $("#item-name").value,
    price: Number($("#item-price").value),
    description: $("#item-desc").value.trim() || null,
    tags: $("#item-tags").value.split(",").map((t) => t.trim()).filter(Boolean),
  };
  try {
    if (editingId) await api(`/items/${editingId}`, { method: "PATCH", body: JSON.stringify(body) });
    else await api("/items", { method: "POST", body: JSON.stringify(body) });
    resetItemForm();
    loadItems();
  } catch (err) {
    $("#item-error").textContent = err.message;   // e.g. validation errors from Pydantic
  }
});
$("#item-cancel").addEventListener("click", resetItemForm);
$("#items-refresh").addEventListener("click", loadItems);
loaders.items = loadItems;

// ------------------------------------------------------------------ titanic

const STATS = [
  {
    path: "/stats/survival-by-class-sex", title: "1. Survival by class and sex", sql: "GROUP BY on two columns",
    columns: [["pclass", "Class"], ["sex", "Sex"], ["passengers", "Passengers", { num: true }],
      ["survivors", "Survivors", { num: true }], ["survival_rate_pct", "Survival %", { num: true }]],
  },
  {
    path: "/stats/survival-by-age-group", title: "2. Survival by age group", sql: "CTE + CASE buckets",
    columns: [["age_group", "Age group"], ["passengers", "Passengers", { num: true }],
      ["survivors", "Survivors", { num: true }], ["survival_rate_pct", "Survival %", { num: true }]],
  },
  {
    path: "/stats/oldest-per-class?top=3", title: "3. Oldest 3 per class", sql: "ROW_NUMBER() OVER (PARTITION BY)",
    columns: [["pclass", "Class"], ["age_rank", "Rank", { num: true }], ["name", "Name"],
      ["age", "Age", { num: true }], ["survived", "Survived", { render: (r) => (r.survived ? "Yes" : "No") }]],
  },
  {
    path: "/stats/survival-by-family-size?min_passengers=5", title: "4. Survival by family size", sql: "GROUP BY expression + HAVING",
    columns: [["family_size", "Family size", { num: true }], ["passengers", "Passengers", { num: true }],
      ["survivors", "Survivors", { num: true }], ["survival_rate_pct", "Survival %", { num: true }]],
  },
  {
    path: "/stats/embarked", title: "5. By port of embarkation", sql: "JOIN + SUM() OVER ()",
    columns: [["port_name", "Port"], ["passengers", "Passengers", { num: true }], ["avg_fare", "Avg fare", { num: true }],
      ["survival_rate_pct", "Survival %", { num: true }], ["share_pct", "Share %", { num: true }]],
  },
];

let statsLoaded = false;

async function loadStats() {
  if (statsLoaded) return;
  statsLoaded = true;
  const container = $("#stats");
  container.replaceChildren(...STATS.map((stat) => {
    const card = el("div", { class: "card" }, el("h2", {}, stat.title), el("p", { class: "hint" }, `SQL: ${stat.sql}`),
      el("div", { class: "table-wrap" }, "Loading…"));
    api(stat.path)
      .then((rows) => card.lastChild.replaceChildren(renderTable(rows, stat.columns)))
      .catch((err) => card.lastChild.replaceChildren(el("p", { class: "form-error" }, err.message)));
    return card;
  }));
}

$("#pass-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const params = new URLSearchParams({ limit: "50" });
  const fields = { pclass: "#pass-class", sex: "#pass-sex", survived: "#pass-survived", name: "#pass-name" };
  for (const [key, selector] of Object.entries(fields)) {
    const value = $(selector).value.trim();
    if (value) params.set(key, value);
  }
  $("#pass-error").textContent = "";
  try {
    const rows = await api(`/passengers?${params}`);
    $("#pass-table").replaceChildren(renderTable(rows, [
      ["passenger_id", "ID", { num: true }], ["name", "Name"], ["pclass", "Class"], ["sex", "Sex"],
      ["age", "Age", { num: true }], ["fare", "Fare", { num: true }],
      ["survived", "Survived", { render: (r) => (r.survived ? "Yes" : "No") }],
    ]));
  } catch (err) {
    $("#pass-error").textContent = err.message;
  }
});
loaders.titanic = loadStats;

// ------------------------------------------------------------------ start

let startTab = "chat";
try { startTab = localStorage.getItem("tab") || "chat"; } catch { /* storage may be blocked */ }
showTab(document.querySelector(`.tab[data-tab="${startTab}"]`) ? startTab : "chat");
loadStatus();
