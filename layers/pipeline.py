# layers/pipeline.py
#
# Chunking ve Analyzing'i aynı anda çalıştırır.
# Chunking bir blok bitirince Analyzer hemen alır — beklemez.
#
# Akış:
#   messages → ChunkingLayer (blokları paralel işler)
#             → Queue (her chunk hazır olunca koyar)
#             → ChunkAnalyzer (queue'dan alır, paralel analiz eder)
#             → sıralı analizler listesi

import asyncio
from typing import List, Dict

from groq import AsyncGroq
import config
import re
import json

# ---- Chunking sabitleri ----
CHUNKING_SYSTEM_PROMPT = """You are a topic-change detector for conversations.
You will receive a numbered list of conversation exchanges.
Your job is to find where the topic significantly changes.
Rules:
- A topic change is when the conversation shifts to a clearly different subject.
- Small topic drifts within the same subject do NOT count as a topic change.
- Return ONLY a JSON array of message numbers where a new topic begins.
- Message 1 is always the start, do NOT include it.
- If there are no topic changes, return an empty array: []
Example response: [6, 12]
Return ONLY the JSON array. No explanation. No markdown. No extra text."""

# ---- Analyzer sabitleri ----
ANALYZER_SYSTEM_PROMPT = """You are a conversation analyst. You will receive a conversation chunk and extract the most important information from it.
Return ONLY a JSON object with exactly these fields:
{
  "topic": "one sentence describing what this conversation is about",
  "decisions": [
    {
      "what": "the decision or solution that was reached",
      "why": "the reason or problem that led to this decision, empty string if not explicitly stated",
      "how": "the implementation or method chosen, empty string if not explicitly stated",
      "failed_attempts": ["what was tried before that did NOT work"]
    }
  ],
  "open_questions": ["list of questions or problems that were raised but not fully resolved"],
  "status": "completed | in_progress | blocked | unknown",
  "keywords": ["important terms, names, numbers, concepts that are central to understanding this conversation"],
  "progress": "one sentence describing what was accomplished or learned in this chunk",
  "context": "any important background information about the user or their project"
}
Rules:
- Be concise. No fluff.
- Preserve all specific numbers, metrics, and technical values exactly as mentioned.
- If a field has nothing relevant, use an empty list [] or empty string "".
- Return ONLY the JSON object. No markdown, no explanation, no extra text.
Primary project detection:
- The conversation may contain multiple topics, side discussions, or examples the user brought up to illustrate a point.
- Identify the PRIMARY project: the one the user is actively building or working on throughout the conversation.
- A topic that appears briefly or is used as an example/test case is NOT the primary project, even if it contains many technical details.
- Decisions and open questions should reflect the primary project. Mention side topics only in "context" and only if they are directly relevant.
- If you are unsure which is the primary project, look for: what the user asks the most questions about, what they are building themselves, what they return to repeatedly."""

MAX_CONCURRENT_CHUNK  = 3
MAX_CONCURRENT_ANALYZE = 3
MAX_RETRIES = 6
BASE_WAIT   = 2

SENTINEL = None  # queue'nun bittiğini işaret eder


class ChunkAnalyzePipeline:
    """
    Chunking ve Analyzing'i pipeline şeklinde çalıştırır.
    Bir chunk hazır olur olmaz Analyzer'a gönderilir.
    """

    def __init__(self, max_tokens_per_block=2000, overlap_messages=3, min_chunk_size=2):
        self.max_tok  = max_tokens_per_block
        self.overlap  = overlap_messages
        self.min_size = min_chunk_size
        self._client  = AsyncGroq(api_key=config.get_groq_key())

    def run(self, messages: List[Dict]) -> List[Dict]:
        """Ana giriş noktası. Sync wrapper."""
        return asyncio.run(self._run_async(messages))

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def _run_async(self, messages: List[Dict]) -> List[Dict]:
        if not messages:
            return []

        blocks = self._build_blocks(messages)
        print(f"[Pipeline] {len(messages)} mesaj → {len(blocks)} blok")

        # Queue: chunking sonuçlarını analyzer'a taşır
        # Her item: (chunk_index, chunk_text) veya SENTINEL
        queue = asyncio.Queue()

        # Önce kaç chunk çıkacağını bilmiyoruz, producer bitince SENTINEL koyar
        chunk_semaphore   = asyncio.Semaphore(MAX_CONCURRENT_CHUNK)
        analyze_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYZE)

        # Sonuçları sırayla toplamak için dict (chunk_index → analiz)
        results: Dict[int, Dict] = {}

        # Producer ve Consumer aynı anda başlar
        await asyncio.gather(
            self._producer(messages, blocks, queue, chunk_semaphore),
            self._consumer(queue, analyze_semaphore, results),
        )

        # chunk_index sırasına göre sırala
        sorted_results = [results[i] for i in sorted(results.keys())]
        return sorted_results

    async def _producer(self, messages, blocks, queue, semaphore):
        """
        Her bloğu paralel olarak LLM'e gönderir.
        Her chunk hazır olunca queue'ya koyar.
        Tüm bloklar bitince SENTINEL koyar.
        """
        chunk_counter = [0]  # mutable sayaç

        async def process_block(block_msgs, offset, idx, total):
            async with semaphore:
                print(f"[Pipeline | Chunking] Blok {idx}/{total} işleniyor...")
                breaks = await self._detect_breaks(block_msgs, idx, total)

                # Break point'lere göre chunk'ları üret
                chunks = self._split_block(block_msgs, offset, breaks, messages)
                for chunk_msgs in chunks:
                    chunk_text = self._msgs_to_text(chunk_msgs)
                    chunk_idx  = chunk_counter[0]
                    chunk_counter[0] += 1
                    await queue.put((chunk_idx, chunk_text))
                    print(f"[Pipeline | Chunking] Chunk {chunk_idx + 1} queue'ya eklendi")

        tasks = [
            process_block(block_msgs, offset, i + 1, len(blocks))
            for i, (block_msgs, offset) in enumerate(blocks)
        ]
        await asyncio.gather(*tasks)

        # Bitti sinyali
        await queue.put(SENTINEL)
        print(f"[Pipeline | Chunking] ✓ Tüm bloklar işlendi, toplam {chunk_counter[0]} chunk")

    async def _consumer(self, queue, semaphore, results):
        """
        Queue'dan chunk'ları alır, paralel analiz eder.
        SENTINEL gelince durur.
        """
        analyze_tasks = []

        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            chunk_idx, chunk_text = item
            # Her chunk için hemen bir analiz task'ı başlat
            task = asyncio.create_task(
                self._analyze_chunk(chunk_idx, chunk_text, semaphore, results)
            )
            analyze_tasks.append(task)

        # Tüm analiz task'larının bitmesini bekle
        if analyze_tasks:
            await asyncio.gather(*analyze_tasks)
        print(f"[Pipeline | Analyzer] ✓ Tüm chunk'lar analiz edildi")

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------

    async def _detect_breaks(self, block_msgs, idx, total) -> List[int]:
        formatted    = self._format_block(block_msgs)
        user_message = f"Find topic changes in this conversation:\n\n{formatted}"

        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(
                    model=config.LAYER_2_MODEL,
                    messages=[
                        {"role": "system", "content": CHUNKING_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content.strip()
                return self._parse_breaks(content)
            except Exception as e:
                if "rate_limit" in str(e) or "429" in str(e):
                    wait = BASE_WAIT ** attempt
                    print(f"[Pipeline | Chunking] Blok {idx} rate limit — {wait}s bekle...")
                    await asyncio.sleep(wait)
                else:
                    print(f"[Pipeline | Chunking] Blok {idx} hata: {e}")
                    return []
        return []

    def _parse_breaks(self, response: str) -> List[int]:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        cleaned  = re.sub(r"```(?:json)?", "", response).replace("```", "").strip()
        match    = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            result = json.loads(match.group())
            return [int(x) for x in result if isinstance(x, (int, float))]
        except (json.JSONDecodeError, ValueError):
            return []

    def _format_block(self, block_msgs) -> str:
        lines = []
        exchange_num = 1
        i = 0
        while i < len(block_msgs):
            msg  = block_msgs[i]
            role = "User" if msg["role"] == "user" else "AI"
            if (msg["role"] == "user"
                    and i + 1 < len(block_msgs)
                    and block_msgs[i+1]["role"] == "assistant"):
                ai_msg = block_msgs[i+1]
                lines.append(f"[{exchange_num}] User: {msg['text']}\n     AI: {ai_msg['text']}")
                i += 2
            else:
                lines.append(f"[{exchange_num}] {role}: {msg['text']}")
                i += 1
            exchange_num += 1
        return "\n\n".join(lines)

    def _split_block(self, block_msgs, offset, exchange_breaks, all_messages):
        """Exchange break noktalarını global mesaj indekslerine çevir, sonra böl."""
        msg_breaks = []
        exchange_num = 1
        msg_idx = 0
        i = 0
        while i < len(block_msgs):
            if exchange_num in exchange_breaks:
                msg_breaks.append(offset + msg_idx)
            if (block_msgs[i]["role"] == "user"
                    and i + 1 < len(block_msgs)
                    and block_msgs[i+1]["role"] == "assistant"):
                msg_idx += 2
                i += 2
            else:
                msg_idx += 1
                i += 1
            exchange_num += 1

        # Bu blok için global mesajları break'lere göre böl
        start = offset
        end   = offset + len(block_msgs) - (self.overlap if offset > 0 else 0)
        block_global = list(range(start, min(end, len(all_messages))))

        if not msg_breaks:
            return [all_messages[start:min(end, len(all_messages))]]

        chunks = []
        prev = start
        for bp in sorted(msg_breaks):
            if start <= bp < end and bp > prev:
                chunks.append(all_messages[prev:bp])
                prev = bp
        chunks.append(all_messages[prev:min(end, len(all_messages))])
        return [c for c in chunks if len(c) >= self.min_size]

    def _msgs_to_text(self, msgs) -> str:
        lines = []
        for msg in msgs:
            role = "User" if msg["role"] == "user" else "AI"
            lines.append(f"{role}: {msg['text']}")
        return "\n".join(lines)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _build_blocks(self, messages):
        blocks = []
        i = 0
        while i < len(messages):
            block = []
            token_count = 0
            block_start = i

            if blocks and self.overlap > 0:
                prev_block_msgs, _ = blocks[-1]
                overlap_msgs = prev_block_msgs[-self.overlap:]
                for m in overlap_msgs:
                    token_count += self._estimate_tokens(m["text"])
                block.extend(overlap_msgs)

            while i < len(messages):
                msg_tokens = self._estimate_tokens(messages[i]["text"])
                new_msgs   = i - block_start
                if token_count + msg_tokens > self.max_tok and new_msgs > 0:
                    break
                block.append(messages[i])
                token_count += msg_tokens
                i += 1

            blocks.append((block, block_start))
        return blocks

    # ------------------------------------------------------------------
    # Analyzer helpers
    # ------------------------------------------------------------------

    async def _analyze_chunk(self, chunk_idx, chunk_text, semaphore, results):
        async with semaphore:
            print(f"[Pipeline | Analyzer] Chunk {chunk_idx + 1} analiz ediliyor...")
            user_message = f"Analyze this conversation chunk:\n\n{chunk_text}"

            for attempt in range(MAX_RETRIES):
                try:
                    response = await self._client.chat.completions.create(
                        model=config.LAYER_3_MODEL,
                        messages=[
                            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        temperature=0,
                    )
                    content = response.choices[0].message.content.strip()
                    results[chunk_idx] = self._parse_analysis(content, chunk_idx + 1)
                    return

                except Exception as e:
                    if "rate_limit" in str(e) or "429" in str(e):
                        wait = BASE_WAIT ** attempt
                        print(f"[Pipeline | Analyzer] Chunk {chunk_idx + 1} rate limit — {wait}s bekle...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"[Pipeline | Analyzer] Chunk {chunk_idx + 1} hata: {e}")
                        results[chunk_idx] = self._empty_analysis(chunk_idx + 1)
                        return

            results[chunk_idx] = self._empty_analysis(chunk_idx + 1)

    def _parse_analysis(self, response: str, chunk_number: int) -> Dict:
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            cleaned = cleaned[start:end]
        try:
            data = json.loads(cleaned)
            context = data.get("context", "")
            print(f"  [Chunk {chunk_number}] context: {len(context)} karakter")  # ← bunu ekle
            return {
                "chunk_number":   chunk_number,
                "topic":          data.get("topic", ""),
                "decisions":      data.get("decisions", []),
                "open_questions": data.get("open_questions", []),
                "progress":       data.get("progress", ""),
                "context":        context,
            }
        except json.JSONDecodeError:
            print(f"[Pipeline | Analyzer] JSON parse hatası (chunk {chunk_number})")
            return self._empty_analysis(chunk_number)

    def _empty_analysis(self, chunk_number: int) -> Dict:
        return {
            "chunk_number":   chunk_number,
            "topic":          "",
            "decisions":      [],
            "open_questions": [],
            "progress":       "",
            "context":        "",
        }