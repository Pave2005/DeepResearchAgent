from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from dotenv import load_dotenv

# Загружает переменные из .env в os.environ
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DeepResearchCore")


# ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ

def extract_json_payload(raw_text: str) -> Any:
    text = raw_text.strip()
    # Удаляем служебные рассуждения thinking-моделей
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate = code_block_match.group(1).strip() if code_block_match else text

    start_curly = candidate.find("{")
    start_square = candidate.find("[")

    starts = [pos for pos in (start_curly, start_square) if pos != -1]
    if not starts:
        raise ValueError(f"В ответе модели отсутствует JSON: {raw_text[:200]}")

    start_pos = min(starts)
    is_object = (start_pos == start_curly)
    end_pos = candidate.rfind("}" if is_object else "]")

    if end_pos == -1 or end_pos < start_pos:
        raise ValueError(f"Некорректная структура JSON: {raw_text[:200]}")

    json_str = candidate[start_pos : end_pos + 1]
    return json.loads(json_str)



@dataclass
class SourceDocument:
    source_id: int
    url: str
    title: str
    snippet: str
    full_content: Optional[str] = None
    domain: str = ""

    def __post_init__(self):
        domain_match = re.search(r"https?://([^/]+)", self.url)
        raw_domain = domain_match.group(1).lower() if domain_match else self.url
        self.domain = re.sub(r"^www\.", "", raw_domain)


@dataclass
class VerifiedFact:
    """Элементарное проверяемое утверждение с привязкой к источникам."""
    fact_id: int
    statement: str
    source_ids: List[int]
    nli_confidence: float
    is_multi_source: bool = False


@dataclass
class ResearchReportPayload:
    """Структура результатов исследования."""
    query: str
    report_text: str
    sources: List[SourceDocument]
    facts: List[VerifiedFact]
    overall_confidence: float
    confidence_breakdown: Dict[str, float]
    executed_subqueries: List[str]


# КЛИЕНТ KEENABLE ПОИСКА

class KeenableClient:

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.keenable.ai",
        use_mcp_transport: bool = False,
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.mcp_endpoint = f"{self.base_url}/mcp"
        self.use_mcp_transport = use_mcp_transport
        self.client_timeout = httpx.Timeout(timeout, read=120.0)
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "User-Agent": "ResearchAgentEngine/2.0",
        }

    async def search(
        self,
        query: str,
        max_results: int = 5,
        site: Optional[str] = None,
        published_after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Исполнение поискового запроса."""
        if self.use_mcp_transport:
            return await self._search_mcp(query, max_results, site, published_after)
        return await self._search_rest(query, max_results, site, published_after)

    async def fetch(
        self,
        url: str,
        max_chars: int = 15000,
        live: bool = False
    ) -> Dict[str, Any]:
        if self.use_mcp_transport:
            return await self._fetch_mcp(url, max_chars, live)
        return await self._fetch_rest(url, max_chars, live)

    async def _search_rest(
        self,
        query: str,
        max_results: int,
        site: Optional[str],
        published_after: Optional[str]
    ) -> List[Dict[str, Any]]:
        endpoint = f"{self.base_url}/v1/search"
        payload: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "snippet_max_length": 600,
        }
        if site:
            payload["site"] = site
        if published_after:
            payload["published_after"] = published_after

        async with httpx.AsyncClient(headers=self.headers, timeout=self.client_timeout) as client:
            try:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json().get("results", [])
            except Exception as exc:
                logger.warning(f"Сбой HTTP REST при поиске '{query}': {exc}")
                return []

    async def _fetch_rest(self, url: str, max_chars: int, live: bool) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/v1/fetch"
        params = {"url": url, "max_chars": max_chars, "live": str(live).lower()}
        async with httpx.AsyncClient(headers=self.headers, timeout=self.client_timeout) as client:
            try:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                logger.warning(f"Сбой HTTP REST при загрузке URL {url}: {exc}")
                return {"url": url, "content": "", "title": "Fetch Error"}

    async def _search_mcp(
        self,
        query: str,
        max_results: int,
        site: Optional[str],
        published_after: Optional[str]
    ) -> List[Dict[str, Any]]:
        args: Dict[str, Any] = {"query": query, "max_results": max_results}
        if site:
            args["site"] = site
        if published_after:
            args["published_after"] = published_after

        for tool_name in ["web_search", "search_web_pages"]:
            rpc_body = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": args}
            }
            async with httpx.AsyncClient(headers=self.headers, timeout=self.client_timeout) as client:
                try:
                    response = await client.post(self.mcp_endpoint, json=rpc_body)
                    response.raise_for_status()
                    data = response.json()
                    if "error" in data:
                        continue

                    content_blocks = data.get("result", {}).get("content", [])
                    for block in content_blocks:
                        if block.get("type") == "text":
                            parsed = json.loads(block.get("text", "{}"))
                            if isinstance(parsed, list):
                                return parsed
                            return parsed.get("results", [])
                except Exception:
                    continue

        return await self._search_rest(query, max_results, site, published_after)

    async def _fetch_mcp(self, url: str, max_chars: int, live: bool) -> Dict[str, Any]:
        rpc_body = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "fetch_page_content",
                "arguments": {"url": url, "max_chars": max_chars, "live": live}
            }
        }
        async with httpx.AsyncClient(headers=self.headers, timeout=self.client_timeout) as client:
            try:
                response = await client.post(self.mcp_endpoint, json=rpc_body)
                response.raise_for_status()
                data = response.json()
                content_blocks = data.get("result", {}).get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        return json.loads(block.get("text", "{}"))
                return {"url": url, "content": ""}
            except Exception as exc:
                return await self._fetch_rest(url, max_chars, live)




class InferenceEngine:
    """Абстракция взаимодействия с моделями через LiteLLM Proxy."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def complete(
        self,
        system_instruction: str,
        user_input: str,
        temperature: float = 0.1,
        enforce_json: bool = False
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_input}
            ],
            "temperature": temperature
        }
        if enforce_json:
            payload["response_format"] = {"type": "json_object"}

        timeout_config = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(headers=self.headers, timeout=timeout_config) as client:
                    response = await client.post(f"{self.base_url}/chat/completions", json=payload)

                    if response.status_code == 429 or (500 <= response.status_code < 600):
                        if attempt < max_attempts:
                            delay = 2 ** attempt
                            logger.warning(f"Шлюз вернул HTTP {response.status_code}. Повтор через {delay}с...")
                            await asyncio.sleep(delay)
                            continue

                    response.raise_for_status()
                    data = response.json()

                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or msg.get("reasoning_content") or ""
                    if not content:
                        if attempt < max_attempts:
                            delay = 2 ** attempt
                            logger.warning(f"Шлюз вернул пустой контент. Повтор {attempt}/{max_attempts} через {delay}с...")
                            await asyncio.sleep(delay)
                            continue
                        raise RuntimeError("Сервер прислал пустой ответ после всех попыток")
                    return content

            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as exc:
                if attempt < max_attempts:
                    delay = 2 ** attempt
                    logger.warning(f"Сбой соединения ({type(exc).__name__}). Повтор {attempt}/{max_attempts} через {delay}с...")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Не удалось восстановить соединение с LiteLLM: {exc}")
                raise
            except httpx.HTTPStatusError as exc:
                logger.error(f"HTTP ошибка LLM шлюза [{exc.response.status_code}]: {exc.response.text}")
                raise RuntimeError(f"Сбой LiteLLM Proxy ({exc.response.status_code}): {exc.response.text}") from exc


# АГЕНТЫ

class QueryPlannerAgent:

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    async def decompose(self, main_query: str) -> List[str]:
        system_prompt = (
            "You are a lead search strategy architect.\n"
            "Decompose a complex research question into 2-3 highly specific sub-queries "
            "for a web search engine. Formulate sub-queries in the primary language of the domain sources (English/Russian).\n"
            "Provide the response STRICTLY as valid JSON:\n"
            '{"subqueries": ["query 1", "query 2"]}'
        )
        user_prompt = f"Research question: {main_query}"
        raw = await self.engine.complete(system_prompt, user_prompt, temperature=0.1, enforce_json=True)
        try:
            parsed = extract_json_payload(raw)
            if isinstance(parsed, dict):
                queries = parsed.get("subqueries", [])
            elif isinstance(parsed, list):
                queries = [q for q in parsed if isinstance(q, str)]
            else:
                queries = []
            return queries if queries else [main_query]
        except Exception as exc:
            logger.warning(f"Failed to parse query decomposition plan ({exc}). Falling back to primary query.")
            return [main_query]


class ContentCollectorAgent:

    def __init__(self, search_client: KeenableClient):
        self.search_client = search_client

    async def execute_gathering(self, queries: List[str], limit_per_query: int = 3) -> List[SourceDocument]:
        documents: List[SourceDocument] = []
        visited_urls = set()
        lock = asyncio.Lock()
        doc_counter = 1

        async def fetch_query(q: str):
            nonlocal doc_counter
            results = await self.search_client.search(q, max_results=limit_per_query)
            for item in results:
                url = item.get("url")
                if not url:
                    continue
                async with lock:
                    if url in visited_urls:
                        continue
                    visited_urls.add(url)
                    doc = SourceDocument(
                        source_id=doc_counter,
                        url=url,
                        title=item.get("title") or "Untitled Document",
                        snippet=item.get("snippet") or item.get("description") or ""
                    )
                    doc_counter += 1
                    documents.append(doc)

        await asyncio.gather(*(fetch_query(q) for q in queries))

        lead_docs = documents[:3]
        async def enrich_content(target_doc: SourceDocument):
            data = await self.search_client.fetch(target_doc.url, max_chars=12000, live=False)
            target_doc.full_content = data.get("content", "")

        await asyncio.gather(*(enrich_content(d) for d in lead_docs))
        return documents


class FactAuditorAgent:

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    async def extract_and_verify(
        self, documents: List[SourceDocument], user_goal: str
    ) -> Tuple[List[VerifiedFact], bool]:
        corpus = "\n\n".join([
            f"[Source #{doc.source_id}] (URL: {doc.url}, Domain: {doc.domain})\n"
            f"Title: {doc.title}\n"
            f"Text:\n{doc.full_content[:4500] if doc.full_content else doc.snippet}"
            for doc in documents
        ])

        system_prompt = (
            "You are a lead factual verification and premise analysis specialist.\n\n"
            "CRITICAL PREMISE VERIFICATION:\n"
            "1. Check if the research question contains a false premise, a fictional entity, "
            "or attributes non-existent properties to an object/event (e.g., 'quantum coprocessor in Intel i9-9900K').\n"
            "2. ESSENTIAL DISTINCTION:\n"
            "   - Do NOT confuse broad, ambiguous, or multi-faceted questions (e.g., multi-aspect ASQA queries) "
            "with questions containing a factual falsehood or false premise.\n"
            "   - Set \"false_premise_detected\": true ONLY if the question asserts a factually false, "
            "impossible, or fabricated premise. If the question is simply open-ended or ambiguous, it is NOT a trap.\n"
            "3. If the question contains a false premise:\n"
            "   - Set \"false_premise_detected\": true.\n"
            "   - In the 'facts' list, MANDATORILY include refutation statements supported by the sources.\n"
            "4. If the question is valid/factual:\n"
            "   - Set \"false_premise_detected\": false.\n\n"
            "Based on the provided sources, extract 3 to 6 atomic, verifiable facts "
            "directly answering the question or directly refuting the false premise.\n"
            "Each fact MUST cite source IDs from the registry.\n\n"
            "Strict JSON format:\n"
            "{\n"
            '  "false_premise_detected": false,\n'
            '  "facts": [\n'
            '    {"statement": "Verifiable factual statement", "sources": [1, 2], "confidence": 0.95}\n'
            "  ]\n"
            "}"
        )
        user_prompt = f"Research Goal: {user_goal}\n\nSource Corpus:\n{corpus}"
        raw = await self.engine.complete(system_prompt, user_prompt, temperature=0.0, enforce_json=True)

        verified_facts: List[VerifiedFact] = []
        is_false_premise = False

        try:
            parsed = extract_json_payload(raw)
            if isinstance(parsed, dict):
                is_false_premise = bool(parsed.get("false_premise_detected", False))
                raw_facts = parsed.get("facts", [])
            elif isinstance(parsed, list):
                is_false_premise = False
                raw_facts = parsed
            else:
                raw_facts = []

            for idx, item in enumerate(raw_facts, start=1):
                if not isinstance(item, dict):
                    continue
                raw_sources = item.get("sources") or []
                src_list = [int(s) for s in raw_sources if str(s).isdigit()]
                try:
                    conf = float(item.get("confidence", 0.8))
                except (ValueError, TypeError):
                    conf = 0.8

                involved_domains = {d.domain for d in documents if d.source_id in src_list}
                is_multi = len(involved_domains) >= 2
                if is_multi:
                    conf = min(0.99, conf + 0.05)

                statement = (item.get("statement") or "").strip()
                if not statement:
                    continue

                verified_facts.append(VerifiedFact(
                    fact_id=idx,
                    statement=statement,
                    source_ids=src_list,
                    nli_confidence=round(conf, 3),
                    is_multi_source=is_multi
                ))
        except Exception as exc:
            logger.error(f"Сбой при разборе фактологической структуры: {exc}. Ответ: {raw[:300]}")

        return verified_facts, is_false_premise


class SynthesisAndCalibrationModule:

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def calculate_confidence(
        self,
        facts: List[VerifiedFact],
        documents: List[SourceDocument]
    ) -> Tuple[float, Dict[str, float]]:
        """Расчет индекса уверенности."""
        if not facts:
            return 0.05, {
                "mean_fact_confidence": 0.0,
                "claim_citation_coverage": 0.0,
                "cross_source_consensus": 0.0,
                "unique_domain_diversity": 0.0
            }

        mean_fact_conf = sum(f.nli_confidence for f in facts) / len(facts)
        cited_facts_count = sum(1 for f in facts if len(f.source_ids) > 0)
        citation_coverage = cited_facts_count / len(facts)
        consensus_ratio = sum(1 for f in facts if f.is_multi_source) / len(facts)

        cited_ids = {sid for f in facts for sid in f.source_ids}
        cited_domains = {doc.domain for doc in documents if doc.source_id in cited_ids}
        domain_diversity = min(1.0, len(cited_domains) / 3.0)

        overall_score = (
            0.40 * mean_fact_conf +
            0.30 * citation_coverage +
            0.20 * consensus_ratio +
            0.10 * domain_diversity
        )

        # Защитная калибровка
        if citation_coverage < 0.50 or len(facts) < 2:
            overall_score = min(overall_score, 0.35)

        calibrated_score = round(max(0.05, min(0.99, overall_score)), 3)

        metrics = {
            "mean_fact_confidence": round(mean_fact_conf, 3),
            "claim_citation_coverage": round(citation_coverage, 3),
            "cross_source_consensus": round(consensus_ratio, 3),
            "unique_domain_diversity": round(domain_diversity, 3),
        }
        return calibrated_score, metrics

    async def generate_report(
        self,
        query: str,
        facts: List[VerifiedFact],
        documents: List[SourceDocument],
        confidence: float
    ) -> str:
        facts_block = "\n".join([
            f"Fact #{f.fact_id} (Supported by sources: {f.source_ids}): {f.statement}"
            for f in facts
        ])
        sources_block = "\n".join([
            f"[{d.source_id}] {d.title} — {d.url}" for d in documents
        ])

        system_prompt = (
            "You are an academic researcher and intelligence analyst.\n"
            "Write a coherent, factually precise, and well-structured report STRICTLY in the same language "
            "as the research question (if the question is in English, write in English; if in Russian, write in Russian).\n\n"
            "The report must rely exclusively on the provided verified facts.\n\n"
            "Citation rules:\n"
            "1. EVERY single sentence in the report MUST end with bracketed source citations "
            "(e.g., [1], [1][2], or [1, 2]), where the numbers strictly match source IDs from the registry.\n"
            "2. Strictly avoid generic conversational intros, greetings, or conclusions lacking citations.\n"
            "3. Do NOT extrapolate or introduce statements absent from the verified data."
        )
        user_prompt = (
            f"Research Question: {query}\n\n"
            f"Verified Facts:\n{facts_block}\n\n"
            f"Source Registry:\n{sources_block}\n\n"
            f"Calculated System Confidence: {confidence * 100:.1f}%\n\n"
            "Generate the structured cited report:"
        )
        return await self.engine.complete(system_prompt, user_prompt, temperature=0.1)




class DeepResearchOrchestrator:

    def __init__(
        self,
        keenable_key: str,
        llm_api_key: str,
        llm_base_url: str,
        model_name: str,
        use_mcp: bool = False
    ):
        self.search_client = KeenableClient(api_key=keenable_key, use_mcp_transport=use_mcp)
        self.engine = InferenceEngine(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model_name=model_name
        )
        self.planner = QueryPlannerAgent(self.engine)
        self.collector = ContentCollectorAgent(self.search_client)
        self.auditor = FactAuditorAgent(self.engine)
        self.synthesizer = SynthesisAndCalibrationModule(self.engine)

    async def run(self, query: str) -> ResearchReportPayload:
        logger.info(f"Запуск глубокого исследования: '{query}'")

        # Декомпозиция запроса
        subqueries = await self.planner.decompose(query)
        logger.info(f"Сформировано поисковых подзапросов: {subqueries}")

        # Сбор документов
        documents = await self.collector.execute_gathering(subqueries)
        logger.info(f"Собрано уникальных веб-документов: {len(documents)}")

        if not documents:
            return ResearchReportPayload(
                query=query,
                report_text="Unable to locate relevant web resources to answer the specified research query.",
                sources=[],
                facts=[],
                overall_confidence=0.05,
                confidence_breakdown={"data_availability": 0.0},
                executed_subqueries=subqueries
            )

        # Извлечение фактов
        facts, is_false_premise = await self.auditor.extract_and_verify(documents, query)
        logger.info(f"Верифицировано фактологических утверждений: {len(facts)}")

        # Расчет уверенности и калибровка
        overall_conf, metrics = self.synthesizer.calculate_confidence(facts, documents)

        if is_false_premise:
            overall_conf = round(min(overall_conf * 0.25, 0.25), 3)
            metrics["false_premise_penalty"] = 0.25
            logger.warning(f"Обнаружена ложная предпосылка! Калиброванная уверенность: {overall_conf * 100:.1f}%")

        if not facts:
            return ResearchReportPayload(
                query=query,
                report_text="Unable to extract verifiable factual claims from the retrieved sources.",
                sources=documents,
                facts=[],
                overall_confidence=overall_conf,
                confidence_breakdown=metrics,
                executed_subqueries=subqueries
            )

        # Синтез
        final_text = await self.synthesizer.generate_report(query, facts, documents, overall_conf)

        return ResearchReportPayload(
            query=query,
            report_text=final_text,
            sources=documents,
            facts=facts,
            overall_confidence=overall_conf,
            confidence_breakdown=metrics,
            executed_subqueries=subqueries
        )


async def main():
    KEENABLE_API_KEY = os.getenv("KEENABLE_API_KEY", "")
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "").rstrip("/")
    MODEL_NAME = os.getenv("MODEL_NAME", "")

    required_vars = {
        "KEENABLE_API_KEY": KEENABLE_API_KEY,
        "LITELLM_API_KEY": LITELLM_API_KEY,
        "LITELLM_BASE_URL": LITELLM_BASE_URL,
        "MODEL_NAME": MODEL_NAME,
    }
    missing_vars = [name for name, value in required_vars.items() if not value]

    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}! "
            "Please configure them in your .env file before running."
        )
        return


    orchestrator = DeepResearchOrchestrator(
        keenable_key=KEENABLE_API_KEY,
        llm_api_key=LITELLM_API_KEY,
        llm_base_url=LITELLM_BASE_URL,
        model_name=MODEL_NAME
    )

    research_topic = (
        "What are the architectural differences between Streamable HTTP and HTTP+SSE transports "
        "in the Model Context Protocol (MCP) specification, and how does the ALCE benchmark "
        "evaluate citation quality in generative models?"
    )

    result = await orchestrator.run(research_topic)

    print("\n" + "=" * 80)
    print("ОТЧЕТ АВТОНОМНОГО ИССЛЕДОВАТЕЛЬСКОГО АГЕНТА")
    print(f"Тема: {result.query}")
    print(f"Калиброванная системная уверенность: {result.overall_confidence * 100:.1f}%")
    print("=" * 80 + "\n")
    print(result.report_text)


if __name__ == "__main__":
    asyncio.run(main())
