import csv
import io
import json
import logging
import re
from typing import Any, Dict, List
import httpx
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetBuilder")


def parse_csv_stream(content: str) -> List[Dict[str, Any]]:
    if content.startswith('\ufeff'):
        content = content[1:]

    if "<html" in content.lower() or "<!doctype" in content.lower():
        logger.error(f"Google вернул HTML-страницу вместо CSV:\n{content[:400]}")
        return []

    raw_rows = list(csv.reader(io.StringIO(content)))
    if not raw_rows:
        logger.error("Полученный CSV файл пуст.")
        return []

    header_idx = -1
    headers = []
    for r_idx, row in enumerate(raw_rows[:15]):
        cleaned = [str(c).strip().lower() for c in row]
        if "question" in cleaned:
            header_idx = r_idx
            headers = cleaned
            logger.info(f"Найдена строка заголовков (строка {header_idx + 1}): {row}")
            break

    if header_idx == -1:
        logger.error("Не удалось найти строку с заголовком 'question'")
        return []

    q_idx = headers.index("question")
    fp_idx = headers.index("false_premise") if "false_premise" in headers else -1
    type_idx = headers.index("fact_type") if "fact_type" in headers else -1

    records = []
    for row in raw_rows[header_idx + 1:]:
        if not row or len(row) <= q_idx:
            continue

        q_text = row[q_idx].strip()
        if not q_text or len(q_text) < 10:
            continue

        is_trap = False
        if fp_idx != -1 and len(row) > fp_idx:
            fp_val = row[fp_idx].strip().lower()
            is_trap = fp_val in ["true", "1", "yes", "t"]

        fact_type = row[type_idx].strip().lower() if (type_idx != -1 and len(row) > type_idx) else "general"
        category = "false_premise" if is_trap else fact_type

        records.append({
            "id": f"FRESHQA_{len(records)+1:04d}",
            "category": category,
            "question": q_text,
            "is_trap": is_trap
        })

    traps_count = sum(1 for r in records if r['is_trap'])
    logger.info(f"Успешно обработано {len(records)} оригинальных вопросов FreshQA (из них ловушек: {traps_count})")
    return records


def fetch_original_freshqa() -> List[Dict[str, Any]]:
    headers = {"User-Agent": "Mozilla/5.0"}

    with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
        readme_url = "https://raw.githubusercontent.com/freshllms/freshqa/main/README.md"
        logger.info(f"Загрузка документации репозитория: {readme_url}")
        resp = client.get(readme_url)
        if resp.status_code != 200:
            resp = client.get("https://raw.githubusercontent.com/freshllms/freshqa/master/README.md")
        resp.raise_for_status()
        readme_text = resp.text

        sheet_match = re.search(r'docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)', readme_text)
        if not sheet_match:
            raise RuntimeError("В README.md freshllms/freshqa не найдена ссылка на Google Sheets.")

        sheet_id = sheet_match.group(1)
        gid_match = re.search(r'gid=(\d+)', readme_text)
        gid_param = f"&gid={gid_match.group(1)}" if gid_match else ""

        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv{gid_param}"
        logger.info(f"Выгрузка оригинального датасета: {export_url}")

        csv_resp = client.get(export_url)
        csv_resp.raise_for_status()

        return parse_csv_stream(csv_resp.text)

def fetch_original_alce_asqa() -> List[Dict[str, Any]]:
    logger.info("Загрузка сплита ASQA (din0s/asqa)...")
    asqa_dataset = load_dataset("din0s/asqa", split="dev")
    records = []

    for idx, item in enumerate(asqa_dataset, start=1):
        q_text = item.get("ambiguous_question", "").strip()
        sample_id = item.get("sample_id", f"{idx:04d}")
        if q_text:
            records.append({
                "id": f"ALCE_ASQA_{sample_id}",
                "category": "multi_aspect",
                "question": q_text,
                "is_trap": False
            })

    logger.info(f"Успешно обработано {len(records)} оригинальных вопросов ALCE-ASQA")
    return records


def main():
    asqa_records = fetch_original_alce_asqa()
    freshqa_records = fetch_original_freshqa()

    os.makedirs("data", exist_ok=True)

    full_dataset = asqa_records + freshqa_records
    full_output = "data/benchmark_dataset_full.json"
    with open(full_output, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"Полный датасет сохранен в '{full_output}' ({len(full_dataset)} вопросов).")

    traps = [r for r in freshqa_records if r["is_trap"]]
    non_traps = [r for r in freshqa_records if not r["is_trap"]]

    asqa_subset = asqa_records[:40]
    freshqa_subset = (traps[:10] + non_traps[:10]) if len(traps) >= 10 else freshqa_records[:20]

    benchmark_60 = asqa_subset + freshqa_subset
    output_path = "data/benchmark_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_60, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Основной датасет '{output_path}' успешно создан: {len(benchmark_60)} сценариев.")
    logger.info(
        f"Состав: ASQA={len(asqa_subset)}, FreshQA={len(freshqa_subset)} "
        f"(из них ловушек: {sum(1 for r in freshqa_subset if r['is_trap'])})"
    )

    smoke_subset = asqa_records[:4] + traps[:2] + non_traps[:2]
    smoke_output_path = "data/benchmark_dataset_quick.json"
    with open(smoke_output_path, "w", encoding="utf-8") as f:
        json.dump(smoke_subset, f, ensure_ascii=False, indent=2)

    logger.info(f"\nЭкспресс-датасет для смоук-теста '{smoke_output_path}' создан: {len(smoke_subset)} сценариев.")
    logger.info(
        f"Состав смоук-теста: ASQA={len(asqa_records[:4])}, "
        f"FreshQA Traps={len(traps[:2])}, FreshQA Normal={len(non_traps[:2])}"
    )
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
