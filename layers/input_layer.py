import re
from typing import List, Dict


class InputParser:
    """
    Farklı AI platformlarından kopyalanan sohbetleri ayrıştırır.

    Desteklenen platformlar:
    - Claude web  : "You said:" / "Claude responded:"
    - ChatGPT     : "You:" / "ChatGPT:"
    - Claude API  : "Human:" / "Assistant:"
    - Gemini      : "Sen:" / "Gemini:"
    - Genel       : "User:" / "AI:"

    Claude web'in özel davranışları:
    1. "You said: <içerik>" → kullanıcı mesajı başlangıcı
    2. Hemen ardından aynı içerik duplicate gelir → SKIP
    3. "Claude responded: <ilk satır>" → AI mesajı başlangıcı
    4. Hemen ardından aynı içerik duplicate gelir → SKIP
    5. Sonraki prefix'siz satırlar AI mesajının gövdesi (yeni "You said:" gelene kadar)
    6. "Show more" gibi UI gürültüleri → SKIP
    """

    SKIP_PATTERNS = [
        r"^\d{1,2}:\d{2}\s*(am|pm)$",
        r"^\d{1,2}:\d{2}$",
        r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}$",
        r"^\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$",
        r"^(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+\d{1,2}$",
        r"^show more$",
        r"^show less$",
        r"^copy$",
        r"^copied!$",
        r"^reading .+ skill$",
        r"^created a file.*$",
        r"^read a file.*$",
        r"^\d+\s*of\s*\d+$",           # "1 of 5" sayfalama
        r"^pastedimage\d*$",            # yapıştırılan görsel placeholder
        r"^pasted$",
        r"^---+$",              # claude-chat-exporter ayracı
        r"^# claude conversation$",  # claude-chat-exporter başlığı
    ]

    USER_PREFIXES = [
        "you said:",
        "you:",
        "human:",
        "sen:",
        "user:",
        "USER:",
        "kullanıcı:",
        "ben:",
    ]

    ASSISTANT_PREFIXES = [
        "claude responded:",
        "chatgpt:",
        "assistant:",
        "gemini:",
        "claude:",
        "ai:",
        "AI:",
        "yapay zeka:",
        "bard:",
        "copilot:",
        "grok:",
        "mistral:",
        "deepseek:",
    ]

    # Claude web UI elementleri — prefix'lere yapışık gelebilir
    UI_NOISE = [
        "Free planUpgrade",
        "Free plan",
        "Upgrade",
        "UpgradePlan",
        "Try Pro",
        "Sign up",
        "Log in",
        "Subscribe",
    ]

    def _pre_clean(self, text: str) -> str:
        """
        Prefix'lere yapışık UI gürültüsünü temizler.
        Örnek: "Free planUpgradeYou said: ..." → "You said: ..."
        """
        # Bilinen UI noise'ları kaldır
        for noise in self.UI_NOISE:
            text = text.replace(noise, "")

        # "You said:" / "Claude responded:" gibi prefix'ler
        # başka kelimeye yapışıksa ayır
        # Örnek: "blahYou said:" → "\nYou said:"
        all_prefixes = self.USER_PREFIXES + self.ASSISTANT_PREFIXES
        for prefix in all_prefixes:
            if prefix.startswith('#'):
                continue
            text = re.sub(rf'(?<!\n)(?<!^)({re.escape(prefix)})', r'\n\1', text, flags=re.IGNORECASE)

        return text

    def parse_raw_text(self, text: str) -> List[Dict[str, str]]:
        """
        Ham metin formatındaki sohbeti ayrıştırır.

        Algoritma:
        - "You said:" / "Claude responded:" prefix'lerini yakalar
        - Her prefix'in hemen ardından gelen duplicate satırı atlar
        - AI mesajının gövdesini (prefix'siz satırlar) doğru mesaja ekler
        - "You said:" gelene kadar devam eden satırlar AI'a aittir

        Returns:
            [{"role": "user"|"assistant", "text": "..."}, ...]
        """
        messages: List[Dict[str, str]] = []
        text = self._pre_clean(text)
        lines = text.strip().split("\n")

        # Son eklenen prefix içeriği — duplicate tespiti için
        last_prefix_content: str | None = None
        # Şu an hangi rol yazıyor: "user" | "assistant" | None
        current_role: str | None = None

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            line_lower = line.lower()

            # --- Gürültü filtresi ---
            if self._should_skip(line_lower):
                continue

            # --- Kullanıcı prefix'i? ---
            user_content = self._match_prefix(line, line_lower, self.USER_PREFIXES)
            if user_content is not None:
                current_role = "user"
                if user_content:  # inline içerik varsa hemen ekle
                    last_prefix_content = user_content
                    messages.append({"role": "user", "text": user_content})
                # Boşsa (## Human: gibi) sadece rol değiştir, sonraki satır eklenecek
                continue

            # --- AI prefix'i? ---
            ai_content = self._match_prefix(line, line_lower, self.ASSISTANT_PREFIXES)
            if ai_content is not None:
                current_role = "assistant"
                if ai_content:  # inline içerik varsa hemen ekle
                    last_prefix_content = ai_content
                    messages.append({"role": "assistant", "text": ai_content})
                # Boşsa (## Claude: gibi) sadece rol değiştir, sonraki satır eklenecek
                continue

            # --- Duplicate satır? (Claude web her prefix içeriğini tekrar yazar) ---
            if last_prefix_content is not None and self._is_duplicate(line, last_prefix_content):
                last_prefix_content = None  # sadece bir kez atla
                continue

            # --- Prefix'siz satır: mevcut role'a ekle ---
            if current_role and messages and messages[-1]["role"] == current_role:
                # Mevcut role'un devam satırı
                messages[-1]["text"] += "\n" + line

            elif current_role:
                # Role belli ama henüz bu role için mesaj yok (## Human:\n sonrası)
                messages.append({"role": current_role, "text": line})

            else:
                # Henüz hiç prefix görülmedi — ilk mesaj kullanıcıdan başlıyordur
                current_role = "user"
                messages.append({"role": "user", "text": line})

        # Metinleri temizle
        return self._clean_messages(messages)

    # ------------------------------------------------------------------
    # Yardımcı metodlar
    # ------------------------------------------------------------------

    def _should_skip(self, lower_line: str) -> bool:
        for pattern in self.SKIP_PATTERNS:
            if re.match(pattern, lower_line):
                return True
        return False

    def _match_prefix(
        self, original_line: str, lower_line: str, prefixes: List[str]
    ) -> str | None:
        for prefix in prefixes:
            if lower_line.startswith(prefix):
                content = original_line[len(prefix):].strip()
                # İçerik boşsa (## Human: gibi ayrı satır formatı) boş string dön
                # Caller bunu "rol değişti, sonraki satırlar bu role ait" olarak yorumlar
                return content  # None yerine "" dönebilir
        return None

    def _is_duplicate(self, line: str, reference: str) -> bool:
        """
        Satır, referans içeriğin başı ile eşleşiyor mu?

        Claude web bazen prefix içeriğinin tamamını değil,
        sadece ilk N karakterini duplicate eder (uzun mesajlarda).
        Bu yüzden 'startswith' ile kontrol ediyoruz.
        """
        # Tam eşleşme
        if line == reference:
            return True
        # Referans, satırın başında mı? (kısa duplicate)
        if reference.startswith(line) and len(line) >= 20:
            return True
        # Satır, referansın başında mı? (uzun mesaj kırpılmış duplicate)
        if line.startswith(reference[:60]) and len(reference) >= 60:
            return True
        return False

    def _clean_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Boş mesajları siler, metinleri trim eder,
        ardışık aynı role sahip mesajları birleştirir.
        """
        cleaned = []
        for msg in messages:
            text = msg["text"].strip()
            if not text:
                continue
            # Ardışık aynı rol → birleştir
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["text"] += "\n" + text
            else:
                cleaned.append({"role": msg["role"], "text": text})
        return cleaned