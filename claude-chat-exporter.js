// Claude Chat Exporter
// Kullanım: F12 → Console → yapıştır → Enter

(function () {

  function getMessages() {
    const userEls = document.querySelectorAll('[class*="font-user-message"]');
    const aiEls   = document.querySelectorAll('[class*="font-claude-response-body"]');

    const tagged = [];

    userEls.forEach(el => {
      const text = el.textContent.trim();
      if (text) tagged.push({ el, role: "USER", text });
    });

    aiEls.forEach(el => {
      const text = el.textContent.trim();
      if (text) tagged.push({ el, role: "AI", text });
    });

    // DOM sırasına göre sırala
    tagged.sort((a, b) =>
      a.el.compareDocumentPosition(b.el) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1
    );

    // Ardışık aynı role'ları birleştir (570 parça → tek mesaj)
    const merged = [];
    for (const item of tagged) {
      if (merged.length > 0 && merged[merged.length - 1].role === item.role) {
        merged[merged.length - 1].text += "\n" + item.text;
      } else {
        merged.push({ role: item.role, text: item.text });
      }
    }

    return merged;
  }

  function getTitle() {
    const el = document.querySelector(
      '[data-testid="chat-title-button"] .truncate, ' +
      'button[data-testid="chat-title-button"] div, h1'
    );
    return el?.textContent?.trim() || "claude-conversation";
  }

  function sanitize(name) {
    return name.replace(/[\/\\:*?"<>|]/g, "-").slice(0, 80);
  }

  function download(content, filename) {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  const messages = getMessages();

  if (messages.length === 0) {
    console.error("Hiç mesaj bulunamadı!");
    return;
  }

  const content  = messages.map(m => `${m.role}: ${m.text}`).join("\n\n");
  const filename = sanitize(getTitle()) + ".txt";

  download(content, filename);
  console.log("✅ " + messages.length + " mesaj indirildi: " + filename);

})();