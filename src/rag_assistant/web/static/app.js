(() => {
  "use strict";

  const API = "/api/v1";
  const el = (id) => document.getElementById(id);
  const state = { user: null, conversations: [], documents: [], current: null, selected: new Set(), cursor: null, retry: null, polling: null, authMode: "login", sending: false, uploading: false, renaming: false };
  async function api(path, options = {}, retried = false) {
    const method = (options.method || "GET").toUpperCase();
    const headers = new Headers(options.headers || {});
    const response = await fetch(`${API}${path}`, { ...options, method, headers, credentials: "same-origin" });
    if (response.status === 401 && !retried && !path.startsWith("/auth/")) {
      try {
        const refreshed = await fetch(`${API}/auth/refresh`, { method: "POST", credentials: "same-origin" });
        if (refreshed.ok) return api(path, options, true);
      } catch (_) { /* Expired session is handled below. */ }
      showAuth();
      throw new Error("نشست شما پایان یافته است. دوباره وارد شوید.");
    }
    if (response.status === 401) {
      showAuth();
      throw new Error("برای ادامه دوباره وارد شوید.");
    }
    if (!response.ok) {
      let detail = `خطای درخواست (${response.status})`;
      let payload = null;
      try { const body = await response.json(); payload = body.detail; detail = typeof body.detail === "string" ? body.detail : body.detail?.error_message || body.detail?.detail || detail; } catch (_) { /* Keep generic message. */ }
      const error = new Error(detail);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    if (response.status === 204) return null;
    const type = response.headers.get("content-type") || "";
    return type.includes("application/json") ? response.json() : response;
  }

  function showAuth(message = "") {
    clearInterval(state.polling);
    state.polling = null;
    el("app-view").hidden = true;
    el("auth-view").hidden = false;
    el("auth-error").textContent = message;
    el("auth-error").hidden = !message;
    el("auth-email").focus();
  }

  function showApp() {
    el("auth-view").hidden = true;
    el("app-view").hidden = false;
  }

  function announce(message, kind = "") {
    const box = el("chat-notice");
    box.textContent = message;
    box.className = `notice ${kind}`.trim();
    box.hidden = !message;
  }

  function text(tag, value, className) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = value == null ? "" : String(value);
    return node;
  }

  function conversationTitle(conversation) {
    return conversation.title || "گفتگوی تازه";
  }

  function appendAnswerText(node, value, citations = [], citationCards = []) {
    const source = String(value || "");
    const markers = /\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g;
    let offset = 0;
    for (const match of source.matchAll(markers)) {
      const index = match.index ?? offset;
      node.append(document.createTextNode(source.slice(offset, index)));
      const numbers = [...match[0].matchAll(/\d+/g)];
      let cursor = 0;
      numbers.forEach((numberMatch, numberIndex) => {
        const number = Number(numberMatch[0]);
        const numberOffset = numberMatch.index ?? cursor;
        node.append(document.createTextNode(match[0].slice(cursor, numberOffset)));
        const citationIndex = number - 1;
        if (number > 0 && citations[citationIndex] && citationCards[citationIndex]) {
          const marker = document.createElement("button");
          marker.type = "button";
          marker.className = "citation-marker";
          marker.dir = "ltr";
          marker.textContent = `[${numberMatch[0]}]`;
          marker.setAttribute("aria-label", `رفتن به ارجاع ${number}`);
          marker.addEventListener("click", () => focusCitation(citationCards[citationIndex]));
          node.append(marker);
        } else {
          node.append(document.createTextNode(numberMatch[0]));
        }
        cursor = numberOffset + numberMatch[0].length;
        if (numberIndex < numbers.length - 1) {
          const commaAt = match[0].indexOf(",", cursor);
          if (commaAt !== -1) { node.append(document.createTextNode(match[0].slice(cursor, commaAt + 1))); cursor = commaAt + 1; }
        }
      });
      node.append(document.createTextNode(match[0].slice(cursor)));
      offset = index + match[0].length;
    }
    node.append(document.createTextNode(source.slice(offset)));
  }

  function latestVersion(document) { return (document.versions || []).reduce((latest, item) => !latest || item.version_number > latest.version_number ? item : latest, null); }
  function statusLabel(version) {
    if (!version) return ["بدون نسخه", ""];
    const labels = { pending: ["در صف پردازش", "processing"], processing: ["در حال پردازش", "processing"], active: ["آماده استفاده", "ready"], failed: ["پردازش ناموفق", "failed"] };
    return labels[version.status] || [version.status, ""];
  }

  async function loadDocuments() {
    const result = await api("/documents/available");
    state.documents = result.items || [];
    renderDocuments();
    updatePolling();
  }

  function renderDocuments() {
    const list = el("document-list");
    list.replaceChildren();
    const query = el("document-search").value.trim().toLocaleLowerCase();
    const filtered = state.documents.filter((doc) => String(doc.title || "").toLocaleLowerCase().includes(query));
    for (const doc of filtered) {
      const card = document.createElement("div"); card.className = "document-card";
      const line = document.createElement("div"); line.className = "document-line";
      const check = document.createElement("input"); check.type = "checkbox"; check.id = `doc-${doc.id}`;
      check.checked = state.selected.has(doc.id);
      const version = latestVersion(doc); const ready = Boolean(doc.active_version_id);
      check.disabled = !ready || !state.current || state.sending;
      check.setAttribute("aria-label", `انتخاب سند ${doc.title}`);
      check.addEventListener("change", () => toggleDocument(doc.id, check.checked));
      const title = text("label", doc.title, "document-title"); title.htmlFor = check.id;
      line.append(check, title); card.append(line);
      const [label, stateClass] = ready ? ["آماده استفاده", "ready"] : statusLabel(version);
      card.append(text("span", label, `document-status ${stateClass}`));
      if (version?.status === "failed" && !doc.active_version_id) {
        if (version.failure_message) card.append(text("span", version.failure_message, "document-status failed"));
        const retry = text("button", "تلاش دوباره", "retry-button"); retry.type = "button";
        retry.addEventListener("click", () => retryVersion(doc, version)); card.append(retry);
      }
      list.append(card);
    }
    if (!filtered.length) list.append(text("p", state.documents.length ? "سندی با این نام پیدا نشد." : "هنوز سندی بارگذاری نشده است.", "sidebar-empty"));
    el("document-hint").textContent = state.documents.length ? "اسناد انتخاب‌شده برای این گفتگو:" : "اسناد فعال گفتگو برای پاسخ‌گویی استفاده می‌شوند.";
    el("selected-count").textContent = `${new Intl.NumberFormat("fa-IR").format(state.selected.size)} سند انتخاب شده`;
  }

  function updatePolling() {
    const pending = state.documents.some((doc) => ["pending", "processing"].includes(latestVersion(doc)?.status));
    if (pending && !state.polling) state.polling = setInterval(() => loadDocuments().catch(() => {}), 2500);
    if (!pending && state.polling) { clearInterval(state.polling); state.polling = null; }
  }

  async function toggleDocument(id, checked) {
    if (checked) state.selected.add(id); else state.selected.delete(id);
    try {
      const docs = await api(`/conversations/${state.current.id}/documents`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document_ids: [...state.selected] }) });
      state.selected = new Set(docs.map((doc) => doc.id));
      renderDocuments();
    } catch (error) { announce(error.message, "error"); await loadCurrent(); }
  }

  async function loadConversations() {
    const page = await api("/conversations?limit=100"); state.conversations = page.items || []; renderConversations();
  }

  function renderConversations() {
    const list = el("conversation-list"); list.replaceChildren();
    const query = el("conversation-search").value.trim().toLocaleLowerCase();
    const filtered = state.conversations.filter((conversation) => conversationTitle(conversation).toLocaleLowerCase().includes(query));
    for (const conversation of filtered) {
      const row = text("div", "", "conversation-row");
      const button = text("button", conversationTitle(conversation), `conversation-item${state.current?.id === conversation.id ? " active" : ""}`);
      button.type = "button"; button.setAttribute("aria-current", state.current?.id === conversation.id ? "page" : "false");
      button.disabled = state.sending;
      button.addEventListener("click", () => openConversation(conversation.id));
      const remove = text("button", "×", "conversation-delete");
      remove.type = "button";
      remove.setAttribute("aria-label", `حذف ${conversationTitle(conversation)}`);
      remove.title = "حذف گفتگو";
      remove.disabled = state.sending && state.current?.id === conversation.id;
      remove.addEventListener("click", () => deleteConversation(conversation.id).catch((error) => announce(error.message, "error")));
      row.append(button, remove); list.append(row);
    }
    if (!filtered.length) list.append(text("p", state.conversations.length ? "گفتگویی با این عنوان پیدا نشد." : "هنوز گفتگویی ندارید.", "sidebar-empty"));
  }

  async function createConversation() {
    const conversation = await api("/conversations", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    state.conversations.unshift(conversation); await openConversation(conversation.id);
  }

  async function deleteConversation(id) {
    const conversation = state.conversations.find((item) => item.id === id);
    if (!conversation || !window.confirm(`حذف گفتگوی «${conversationTitle(conversation)}»؟`)) return;
    await api(`/conversations/${id}`, { method: "DELETE" });
    state.conversations = state.conversations.filter((item) => item.id !== id);
    if (state.current?.id === id) {
      state.current = null;
      state.selected.clear();
      el("conversation-view").hidden = true;
      el("rename-conversation").hidden = true;
      el("welcome").hidden = false;
      el("message-list").replaceChildren();
      if (state.conversations.length) await openConversation(state.conversations[0].id);
      else renderDocuments();
    }
    renderConversations();
    announce("گفتگو حذف شد.", "success");
  }

  async function openConversation(id) {
    state.current = await api(`/conversations/${id}`);
    state.selected = new Set(state.current.selected_documents.map((doc) => doc.id));
    state.cursor = null; state.retry = null;
    el("welcome").hidden = true; el("conversation-view").hidden = false;
    el("active-title").textContent = conversationTitle(state.current);
    el("rename-conversation").hidden = false;
    el("message-list").replaceChildren(); announce("");
    const draft = sessionStorage.getItem(draftKey(id)) || "";
    el("question-input").value = draft;
    resizeComposer(); updateComposer();
    await loadHistory(false); renderConversations(); renderDocuments();
    closeSidebar(); el("question-input").focus();
  }

  async function loadCurrent() {
    if (state.current) {
      state.current = await api(`/conversations/${state.current.id}`);
      state.selected = new Set(state.current.selected_documents.map((doc) => doc.id)); renderDocuments();
    }
  }

  async function loadHistory(older) {
    if (!state.current) return;
    const query = new URLSearchParams({ limit: "50" }); if (older && state.cursor) query.set("cursor", state.cursor);
    const page = await api(`/conversations/${state.current.id}/messages?${query}`);
    state.cursor = page.next_cursor; el("older-messages").hidden = !state.cursor;
    const chronological = [...(page.items || [])].reverse();
    const target = el("message-list");
    if (older) {
      const beforeHeight = target.scrollHeight;
      const fragment = document.createDocumentFragment(); chronological.forEach((message) => fragment.append(renderMessage(message)));
      target.prepend(fragment); target.scrollTop = target.scrollHeight - beforeHeight;
    } else {
      const fragment = document.createDocumentFragment(); chronological.forEach((message) => fragment.append(renderMessage(message)));
      target.append(fragment); scrollMessagesToLatest(target);
    }
  }

  function citationParagraph(paragraph, cited) {
    const block = document.createElement("div");
    block.className = `citation-paragraph${cited ? " is-cited" : ""}`;
    const location = `صفحه ${paragraph.page_number} · بند ${paragraph.paragraph_number}`;
    block.append(text("div", location, "citation-meta"));
    const tag = paragraph.kind === "heading" ? "h4" : "p";
    block.append(text(tag, paragraph.original_text, "citation-context-text"));
    return block;
  }

  function renderCitation(citation, index, messageId, conversationId) {
    const details = document.createElement("details");
    details.className = "citation"; details.id = `citation-${crypto.randomUUID()}`;
    const summary = document.createElement("summary");
    summary.append(text("span", index + 1, "citation-number"));
    const description = document.createElement("span"); description.className = "citation-description";
    description.append(text("span", citation.document_title, "citation-title"));
    const location = citation.chunk_id
      ? citation.page_start === citation.page_end
        ? `صفحه ${citation.page_start}`
        : `صفحه ${citation.page_start}–${citation.page_end}`
      : `صفحه ${citation.page_number} · بند ${citation.paragraph_number}`;
    description.append(text("span", location, "citation-location"));
    const citationText = citation.chunk_text || citation.original_text || "";
    const preview = String(citationText).replace(/\s+/g, " ").trim();
    description.append(text("span", preview.length > 120 ? `${preview.slice(0, 120)}…` : preview, "citation-preview"));
    summary.append(description);
    const body = document.createElement("div"); body.className = "citation-body";
    const section = document.createElement("div"); section.className = "citation-section";
    if (citation.chunk_id) {
      const block = document.createElement("div");
      block.className = "citation-paragraph is-cited";
      block.append(text("div", location, "citation-meta"));
      block.append(text("p", citation.chunk_text, "citation-context-text"));
      section.append(block);
    } else {
      section.append(citationParagraph(citation, true));
    }
    const status = text("div", "", "citation-context-status"); status.setAttribute("role", "status");
    const copy = text("button", "کپی متن مورد استناد", "copy-button citation-copy"); copy.type = "button";
    copy.addEventListener("click", () => copyText(citationText, copy));
    body.append(section, status, copy); details.append(summary, body);
    let loaded = false; let loading = false;
    details.addEventListener("toggle", async () => {
      if (citation.chunk_id || !details.open || loaded || loading || !messageId || !conversationId) return;
      loading = true; section.setAttribute("aria-busy", "true"); status.textContent = "در حال دریافت متن پیرامون…";
      try {
        const context = await api(`/conversations/${conversationId}/messages/${messageId}/citations/${citation.source_paragraph_id}/context`);
        // Keep the immutable snapshot visible if the context does not match this citation.
        const cited = context.paragraphs.find((paragraph) => paragraph.is_cited);
        if (!cited || cited.source_paragraph_id !== citation.source_paragraph_id || cited.original_text !== citation.original_text) throw new Error("Citation context mismatch");
        section.replaceChildren(...context.paragraphs.map((paragraph) => citationParagraph(paragraph, paragraph.is_cited)));
        status.textContent = context.truncated ? "برای خوانایی، فقط بندهای نزدیک به ارجاع نمایش داده شده‌اند." : "";
        loaded = true;
        if (details.open) section.querySelector(".is-cited")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      } catch (_) {
        status.textContent = "متن پیرامون در دسترس نیست. متن دقیق ارجاع در بالا نمایش داده شده است. برای تلاش دوباره، ارجاع را ببندید و باز کنید.";
      } finally { loading = false; section.removeAttribute("aria-busy"); }
    });
    return details;
  }

  function renderMessage(message) {
    const bubble = document.createElement("article"); bubble.className = `message ${message.role === "user" ? "user" : "assistant"}`;
    bubble.append(text("div", message.role === "user" ? "شما" : "دستیار", "message-label"));
    if (message.execution_status === "failed") {
      const failure = document.createElement("div"); failure.className = "failure-card";
      failure.append(text("strong", "اجرای پاسخ ناموفق بود"));
      failure.append(text("span", message.error_message || "بازیابی یا تولید پاسخ با خطا روبه‌رو شد. این وضعیت با کمبود شواهد متفاوت است."));
      bubble.append(failure); return bubble;
    }
    const outcome = message.answer_outcome;
    if (outcome) {
      const labels = { fully_answered: "پاسخ بر پایه شواهد", partially_answered: "پاسخ نسبی · بخشی از پرسش بی‌پاسخ است", abstained: "شواهد کافی پیدا نشد" };
      bubble.append(text("span", labels[outcome] || outcome, `outcome${outcome === "partially_answered" ? " partial" : outcome === "abstained" ? " abstained" : ""}`));
    }
    const citationCards = [];
    let citations = null;
    if (message.citations?.length) {
      citations = document.createElement("div"); citations.className = "citation-list";
      message.citations.forEach((citation, index) => {
        const details = renderCitation(citation, index, message.id, state.current?.id);
        citationCards.push(details); citations.append(details);
      });
    }
    const content = text("div", "", "message-content");
    const answerText = message.content || (message.execution_status === "pending" ? "در حال آماده‌سازی پاسخ…" : "");
    appendAnswerText(content, answerText, message.citations || [], citationCards);
    bubble.append(content);
    if (message.role !== "user" && message.content) {
      const copy = text("button", "کپی پاسخ", "copy-button"); copy.type = "button"; copy.addEventListener("click", () => copyText(message.content, copy)); bubble.append(copy);
    }
    if (citations) bubble.append(citations);
    return bubble;
  }

  async function submitQuestion(question, idempotencyKey = crypto.randomUUID(), failedBubble = null) {
    if (state.sending || !state.current) return;
    const conversationId = state.current.id;
    const input = el("question-input");
    if (failedBubble && input.value.trim() === question) input.value = "";
    state.sending = true; updateComposer(); renderDocuments(); renderConversations();
    const target = el("message-list");
    const waiting = renderMessage({ role: "assistant", execution_status: "pending" });
    if (failedBubble) failedBubble.replaceWith(waiting);
    else {
      target.append(renderMessage({ role: "user", content: question }));
      target.append(waiting);
    }
    scrollMessagesToLatest(target);
    try {
      const answer = await api(`/conversations/${conversationId}/messages`, { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": idempotencyKey }, body: JSON.stringify({ content: question }) });
      waiting.replaceWith(renderMessage(answer)); state.retry = null; sessionStorage.removeItem(draftKey(conversationId)); input.value = ""; announce("");
    } catch (error) {
      state.retry = { question, idempotencyKey };
      const pending = error.payload?.execution_status === "pending";
      const failure = pending
        ? renderMessage({ role: "assistant", execution_status: "pending", content: "پاسخ هنوز در حال اجراست. برای دریافت نتیجه دوباره بررسی کنید." })
        : renderMessage(error.payload?.execution_status === "failed" ? error.payload : { role: "assistant", execution_status: "failed", error_message: error.message });
      waiting.replaceWith(failure);
      const retry = text("button", pending ? "بررسی دوباره همین پرسش" : "تلاش دوباره برای همین پرسش", "retry-button"); retry.type = "button"; retry.addEventListener("click", () => submitQuestion(state.retry.question, state.retry.idempotencyKey, failure)); failure.append(retry);
      announce(pending ? "درخواست هنوز در حال اجراست؛ بررسی دوباره از همان شناسه درخواست استفاده می‌کند." : "اجرای پاسخ کامل نشد. در صورت تلاش دوباره، همان کلید درخواست استفاده می‌شود.", pending ? "" : "error");
    } finally { state.sending = false; if (state.retry && input.value === "") { input.value = question; sessionStorage.setItem(draftKey(conversationId), question); } resizeComposer(); updateComposer(); renderDocuments(); renderConversations(); scrollMessagesToLatest(target); }
  }

  function scrollMessagesToLatest(target) {
    requestAnimationFrame(() => {
      target.style.scrollBehavior = "auto";
      target.scrollTop = target.scrollHeight;
      requestAnimationFrame(() => target.style.removeProperty("scroll-behavior"));
    });
  }

  function draftKey(id) { return `rag-draft:${id}`; }
  function resizeComposer() { const input = el("question-input"); input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 170)}px`; }
  function updateComposer() {
    const input = el("question-input"); const button = el("chat-form").querySelector("button[type=submit]");
    input.disabled = state.sending;
    el("chat-form").setAttribute("aria-busy", String(state.sending));
    button.disabled = state.sending || !input.value.trim() || !state.current;
    button.textContent = state.sending ? "در حال ارسال…" : "ارسال";
    el("new-conversation").disabled = state.sending;
    el("welcome-new").disabled = state.sending;
    const count = el("character-count");
    count.hidden = input.value.length < 9000;
    count.textContent = `${new Intl.NumberFormat("fa-IR").format(input.value.length)} / ۱۰٬۰۰۰`;
    count.setAttribute("aria-label", `تعداد نویسه ${input.value.length} از ۱۰۰۰۰`);
  }

  function focusCitation(card) {
    card.open = true;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("citation-highlight");
    window.setTimeout(() => card.classList.remove("citation-highlight"), 1800);
    card.querySelector("summary")?.focus({ preventScroll: true });
  }

  async function copyText(value, button) {
    const original = button.textContent;
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(String(value));
      else {
        const field = document.createElement("textarea"); field.value = String(value); field.setAttribute("readonly", "");
        field.style.position = "fixed"; field.style.opacity = "0"; document.body.append(field); field.select();
        const copied = document.execCommand("copy"); field.remove(); if (!copied) throw new Error("copy failed");
      }
      button.textContent = "کپی شد"; button.classList.add("copy-feedback");
    } catch (_) { button.textContent = "کپی نشد"; button.classList.add("copy-failed"); }
    window.setTimeout(() => { button.textContent = original; button.classList.remove("copy-feedback", "copy-failed"); }, 1800);
  }

  async function uploadPdf(file) {
    if (state.uploading) return;
    if (!file || !file.name.toLowerCase().endsWith(".pdf") || (file.type && file.type !== "application/pdf")) {
      setUploadStatus("فقط فایل PDF پذیرفته می‌شود.", "error"); return;
    }
    state.uploading = true; el("upload-zone").classList.add("uploading");
    setUploadStatus(`در حال بارگذاری ${file.name}…`, "busy");
    const query = new URLSearchParams({ title: file.name.replace(/\.pdf$/i, "") || "سند تازه" });
    try {
      await api(`/documents?${query}`, { method: "POST", headers: { "Content-Type": "application/pdf" }, body: file });
      setUploadStatus(`${file.name} دریافت شد؛ پردازش آن پیگیری می‌شود.`, "success"); await loadDocuments();
    } catch (error) { setUploadStatus(error.message, "error"); }
    finally { state.uploading = false; el("upload-zone").classList.remove("uploading"); }
  }

  function setUploadStatus(message, kind = "") { const status = el("upload-status"); status.textContent = message; status.className = `upload-status ${kind}`; status.hidden = !message; }

  async function retryVersion(doc, version) {
    try { await api(`/documents/${doc.id}/versions/${version.id}/retry`, { method: "POST" }); announce("درخواست پردازش دوباره ثبت شد.", "success"); await loadDocuments(); }
    catch (error) { announce(error.message, "error"); }
  }

  function closeSidebar() { el("sidebar").classList.remove("open"); el("sidebar-backdrop").hidden = true; el("menu-toggle").setAttribute("aria-expanded", "false"); }

  el("auth-form").addEventListener("submit", async (event) => {
    event.preventDefault(); const form = new FormData(event.currentTarget);
    const button = el("auth-submit"); button.disabled = true;
    try {
      const mode = state.authMode === "register" ? "register" : "login";
      state.user = await api(`/auth/${mode}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email: form.get("email"), password: form.get("password") }) });
      await startApp();
    } catch (error) { el("auth-error").textContent = error.message; el("auth-error").hidden = false; }
    finally { button.disabled = false; }
  });
  document.querySelectorAll("[data-auth-mode]").forEach((button) => button.addEventListener("click", () => {
    state.authMode = button.dataset.authMode; document.querySelectorAll("[data-auth-mode]").forEach((item) => item.classList.toggle("active", item === button));
    el("auth-title").textContent = state.authMode === "register" ? "حساب تازه بسازید" : "به فضای گفت‌وگو خوش آمدید";
    el("auth-submit").textContent = state.authMode === "register" ? "ساخت حساب" : "ورود";
    el("auth-password").autocomplete = state.authMode === "register" ? "new-password" : "current-password";
  }));
  el("logout-button").addEventListener("click", async () => { try { await api("/auth/logout", { method: "POST" }); } catch (_) { /* Clear the interface even when the session expired. */ } state.user = null; showAuth(); });
  el("new-conversation").addEventListener("click", () => createConversation().catch((error) => announce(error.message, "error")));
  el("welcome-new").addEventListener("click", () => createConversation().catch((error) => announce(error.message, "error")));
  el("older-messages").addEventListener("click", () => loadHistory(true).catch((error) => announce(error.message, "error")));
  el("rename-conversation").addEventListener("click", () => {
    if (!state.current) return;
    el("rename-input").value = conversationTitle(state.current);
    el("rename-error").hidden = true;
    el("rename-dialog").showModal();
    el("rename-input").focus();
    el("rename-input").select();
  });
  el("rename-cancel").addEventListener("click", () => el("rename-dialog").close());
  el("rename-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const title = el("rename-input").value.trim();
    if (state.renaming || !title || !state.current) return;
    const conversationId = state.current.id;
    state.renaming = true;
    const submit = el("rename-form").querySelector("button[type=submit]");
    submit.disabled = true;
    el("rename-input").disabled = true;
    el("rename-cancel").disabled = true;
    api(`/conversations/${conversationId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }).then((updated) => {
      state.current.title = updated.title;
      state.current.updated_at = updated.updated_at;
      const item = state.conversations.find((conversation) => conversation.id === conversationId);
      if (item) { item.title = updated.title; item.updated_at = updated.updated_at; }
      state.conversations.sort((left, right) => right.updated_at.localeCompare(left.updated_at) || right.id.localeCompare(left.id));
      el("active-title").textContent = updated.title;
      renderConversations();
      el("rename-dialog").close();
      announce("نام گفتگو ذخیره شد.", "success");
    }).catch((error) => {
      el("rename-error").textContent = error.message;
      el("rename-error").hidden = false;
    }).finally(() => {
      state.renaming = false;
      submit.disabled = false;
      el("rename-input").disabled = false;
      el("rename-cancel").disabled = false;
    });
  });
  el("chat-form").addEventListener("submit", (event) => {
    event.preventDefault(); const input = el("question-input"); const question = input.value.trim();
    if (!question || !state.current) return;
    input.value = ""; submitQuestion(question).catch((error) => announce(error.message, "error"));
  });
  el("question-input").addEventListener("input", () => {
    if (state.current) sessionStorage.setItem(draftKey(state.current.id), el("question-input").value);
    resizeComposer(); updateComposer();
  });
  el("question-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!el("chat-form").querySelector("button[type=submit]").disabled) el("chat-form").requestSubmit(); }
  });
  el("conversation-search").addEventListener("input", renderConversations);
  el("document-search").addEventListener("input", renderDocuments);
  el("pdf-file").addEventListener("change", async (event) => {
    const file = event.target.files?.[0]; if (!file) return;
    await uploadPdf(file); event.target.value = "";
  });
  const uploadZone = el("upload-zone");
  uploadZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); el("pdf-file").click(); } });
  ["dragenter", "dragover"].forEach((name) => uploadZone.addEventListener(name, (event) => { event.preventDefault(); uploadZone.classList.add("drag-over"); }));
  ["dragleave", "drop"].forEach((name) => uploadZone.addEventListener(name, (event) => { event.preventDefault(); uploadZone.classList.remove("drag-over"); }));
  uploadZone.addEventListener("drop", (event) => { const file = event.dataTransfer?.files?.[0]; if (file) uploadPdf(file); });
  el("menu-toggle").addEventListener("click", () => { const open = el("sidebar").classList.toggle("open"); el("sidebar-backdrop").hidden = !open; el("menu-toggle").setAttribute("aria-expanded", String(open)); });
  el("sidebar-backdrop").addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && el("sidebar").classList.contains("open")) { closeSidebar(); el("menu-toggle").focus(); } });

  async function startApp() {
    showApp(); el("user-email").textContent = state.user?.email || "";
    await Promise.all([loadConversations(), loadDocuments()]); await loadCurrent();
    if (state.conversations.length) await openConversation(state.conversations[0].id);
    else { el("conversation-view").hidden = true; el("welcome").hidden = false; el("rename-conversation").hidden = true; }
  }

  (async function restoreSession() {
    try { state.user = await api("/auth/me"); await startApp(); }
    catch (_) { showAuth(); }
  })();
})();
