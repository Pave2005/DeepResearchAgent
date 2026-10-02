from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

from dr_system import DeepResearchOrchestrator, ResearchReportPayload

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Evaluator")


# НЕЗАВИСИМЫЙ LLM-СУДЬЯ

class CitationEvaluatorJudge:
    """Независимый оценщик на базе LiteLLM, реализующий NLI-проверку цитирования."""

    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timeout_config = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        self.client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout_config)
        return self.client

    async def close(self):
        if self.client and not self.client.is_closed:
            await self.client.aclose()

    async def evaluate_claim_grounding(self, claim: str, source_text: str) -> bool:
        """Проверка следования ли утверждение из текста."""
        if not claim.strip() or not source_text.strip():
            return False

        prompt = (
            "You are a strict factual consistency and Natural Language Inference (NLI) judge.\n"
            "Determine whether the following CLAIM is fully supported by the text of the SOURCE document.\n"
            "Respond STRICTLY with one word: YES (if fully entailed and supported) or NO (if not supported, "
            "contradicted, or absent from the source).\n\n"
            f"SOURCE:\n{source_text[:4500]}\n\n"
            f"CLAIM:\n{claim}\n\n"
            "VERDICT (YES/NO):"
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 350
        }

        client = await self._get_client()
        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
                if resp.status_code == 429 or (500 <= resp.status_code < 600):
                    if attempt < max_attempts:
                        await asyncio.sleep(2 ** attempt)
                        continue

                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                raw_content = msg.get("content") or msg.get("reasoning_content") or ""

                if not raw_content.strip():
                    if attempt < max_attempts:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False

                cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_content, flags=re.IGNORECASE).strip()
                ans = cleaned.upper()

                verdict_match = re.search(r'\bVERDICT\s*:\s*(YES|NO)\b', ans)
                if verdict_match:
                    return verdict_match.group(1) == "YES"

                first_word = re.search(r'\b(YES|NO)\b', ans[:40])
                if first_word:
                    return first_word.group(1) == "YES"

                yes_matches = list(re.finditer(r'\bYES\b', ans))
                no_matches = list(re.finditer(r'\bNO\b', ans))

                if yes_matches and not no_matches:
                    return True
                if no_matches and not yes_matches:
                    return False
                if yes_matches and no_matches:
                    return yes_matches[0].start() < no_matches[0].start()

                return False

            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError):
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
            except Exception as e:
                logger.warning(f"Ошибка LLM-судьи: {e}")
                return False
        return False


# РАСЧЕТ МЕТРИК

@dataclass
class TestResult:
    test_id: str
    category: str
    question: str
    system_confidence: float
    citation_precision: float
    citation_recall: float
    total_statements: int
    cited_statements: int
    verified_citations: int
    is_trap: bool
    passed_calibration: bool


def split_sentences_safely(text: str) -> List[str]:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    protected = re.sub(r'\b(e\.g|i\.e|vs|etc|fig|dr|mr|ms|prof)\.\s+', r'\1_dot_ ', cleaned, flags=re.IGNORECASE)
    parts = re.split(r'(?<=[.!?])\s+|\n+', protected)

    sentences = []
    for p in parts:
        p_str = p.strip()
        if p_str.startswith("#") and not re.search(r'\[\d+\]', p_str):
            continue
        if len(p_str) > 10:
            sentences.append(p_str.replace('_dot_', '.'))
    return sentences


async def run_evaluation_suite(
    orchestrator: DeepResearchOrchestrator,
    judge: CitationEvaluatorJudge,
    dataset: List[Dict[str, Any]],
    output_filepath: str = "benchmark_run_latest.jsonl"
) -> List[TestResult]:
    results: List[TestResult] = []
    total_tests = len(dataset)
    logger.info(f"Старт бенчмаркинга: {total_tests} сценариев. Лог в {output_filepath}")

    for idx, test_case in enumerate(dataset, start=1):
        t_id = test_case.get("id", f"TEST_{idx}")
        t_cat = test_case.get("category", "general")
        t_q = test_case.get("question", "")
        is_trap = test_case.get("is_trap", False)

        logger.info(f"\n{'='*70}\n[Тест {idx}/{total_tests}] [{t_id}]: {t_q}\n{'='*70}")
        start_time = datetime.now()

        try:
            payload: ResearchReportPayload = await orchestrator.run(t_q)
            sentences = split_sentences_safely(payload.report_text)
            source_map = {doc.source_id: (doc.full_content if doc.full_content else doc.snippet) for doc in payload.sources}

            total_citations = 0
            supported_citations = 0
            cited_sentences_count = 0

            for sent in sentences:
                raw_citations = [
                    int(num)
                    for match in re.findall(r'\[([^\]]+)\]', sent)
                    for num in re.findall(r'\b\d+\b', match)
                ]
                citations = list(dict.fromkeys(raw_citations))

                if citations:
                    cited_sentences_count += 1
                    for cid in citations:
                        total_citations += 1
                        source_context = source_map.get(cid, "")
                        if source_context:
                            clean_sent = re.sub(r'\[[^\]]+\]', '', sent).strip()
                            if await judge.evaluate_claim_grounding(clean_sent, source_context):
                                supported_citations += 1

            precision = (supported_citations / total_citations) if total_citations > 0 else 0.0
            recall = (cited_sentences_count / len(sentences)) if sentences else 0.0

            if is_trap:
                passed_calibration = (payload.overall_confidence < 0.40)
            else:
                passed_calibration = (payload.overall_confidence >= 0.60 and precision >= 0.60)

            result = TestResult(
                test_id=t_id,
                category=t_cat,
                question=t_q,
                system_confidence=payload.overall_confidence,
                citation_precision=round(precision, 3),
                citation_recall=round(recall, 3),
                total_statements=len(sentences),
                cited_statements=cited_sentences_count,
                verified_citations=supported_citations,
                is_trap=is_trap,
                passed_calibration=passed_calibration
            )
        except Exception as exc:
            logger.error(f"Сбой выполнения [{t_id}]: {exc}", exc_info=True)
            result = TestResult(
                test_id=t_id,
                category=t_cat,
                question=t_q,
                system_confidence=0.0,
                citation_precision=0.0,
                citation_recall=0.0,
                total_statements=0,
                cited_statements=0,
                verified_citations=0,
                is_trap=is_trap,
                passed_calibration=False
            )

        results.append(result)
        with open(output_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Завершен [{t_id}] за {elapsed:.1f}с: Conf={result.system_confidence*100:.1f}%, "
            f"Precision={result.citation_precision*100:.1f}%, Recall={result.citation_recall*100:.1f}%, "
            f"Калибровка={'✓' if result.passed_calibration else '✗'}"
        )
        await asyncio.sleep(1.0)

    return results


async def main():
    KEENABLE_API_KEY = os.getenv("KEENABLE_API_KEY", "")
    LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
    LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "").rstrip("/")
    MODEL_NAME = os.getenv("MODEL_NAME", "")
    DATASET_PATH = os.getenv("BENCHMARK_DATASET_PATH", "data/benchmark_dataset.json")

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

    if os.path.exists(DATASET_PATH):
        logger.info(f"Загрузка датасета из внешнего файла: {DATASET_PATH}")
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        raise FileNotFoundError(f"Файл датасета '{DATASET_PATH}' не найден! Запустите prepare_dataset.py.")

    orchestrator = DeepResearchOrchestrator(
        keenable_key=KEENABLE_API_KEY,
        llm_api_key=LITELLM_API_KEY,
        llm_base_url=LITELLM_BASE_URL,
        model_name=MODEL_NAME
    )

    judge = CitationEvaluatorJudge(
        api_key=LITELLM_API_KEY,
        base_url=LITELLM_BASE_URL,
        model_name=MODEL_NAME
    )

    os.makedirs("results", exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/benchmark_results_{run_id}.json"
    jsonl_log_file = f"results/benchmark_run_{run_id}.jsonl"

    logger.info(f"Старт бенчмаркинга: {len(dataset)} сценариев...")

    try:
        results = await run_evaluation_suite(
            orchestrator=orchestrator,
            judge=judge,
            dataset=dataset,
            output_filepath=jsonl_log_file
        )

        print("\n" + "=" * 90)
        print("ИТОГОВАЯ ВЕДОМОСТЬ ОЦЕНКИ ТОЧНОСТИ И АТРИБУЦИИ")
        print("=" * 90)
        print(f"{'ID Теста':<16} | {'Категория':<14} | {'Conf':<6} | {'Precision':<10} | {'Recall':<8} | {'Статус калибровки'}")
        print("-" * 90)

        for r in results:
            calib_str = "✓ PASSED" if r.passed_calibration else "✗ FAILED"
            print(f"{r.test_id:<16} | {r.category:<14} | {r.system_confidence*100:>4.1f}% | {r.citation_precision*100:>8.1f}% | {r.citation_recall*100:>6.1f}% | {calib_str}")

        standard_tests = [r for r in results if not r.is_trap]
        trap_tests = [r for r in results if r.is_trap]

        print("-" * 90)
        if standard_tests:
            avg_precision = sum(r.citation_precision for r in standard_tests) / len(standard_tests)
            avg_recall = sum(r.citation_recall for r in standard_tests) / len(standard_tests)
            avg_conf = sum(r.system_confidence for r in standard_tests) / len(standard_tests)
            print(f"Стандартные тесты (n={len(standard_tests)}):")
            print(f"  • Средняя точность цитирования (Precision): {avg_precision * 100:.1f}%")
            print(f"  • Средняя полнота цитирования (Recall):       {avg_recall * 100:.1f}%")
            print(f"  • Средняя уверенность:                       {avg_conf * 100:.1f}%")

        if trap_tests:
            passed_traps = sum(1 for r in trap_tests if r.passed_calibration)
            trap_success_rate = (passed_traps / len(trap_tests)) * 100
            print(f"Тесты-ловушки (FreshQA Traps, n={len(trap_tests)}):")
            print(f"  • Успешность распознавания и калибровки:     {trap_success_rate:.1f}% ({passed_traps}/{len(trap_tests)})")

        overall_passed = sum(1 for r in results if r.passed_calibration)
        total_count = len(results) if results else 1
        print(f"Общий результат калибровки по выборке:          {overall_passed}/{len(results)} ({(overall_passed/total_count)*100:.1f}%)")
        print("=" * 90)

        summary_data = {
            "run_id": run_id,
            "model": MODEL_NAME,
            "total_tests": len(results),
            "overall_passed_calibration": overall_passed,
            "results": [r.__dict__ for r in results]
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Полные результаты сохранены в файл: {output_file}\n")

    finally:
        await judge.close()


if __name__ == "__main__":
    asyncio.run(main())
