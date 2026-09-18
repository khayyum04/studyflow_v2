from __future__ import annotations

import base64
import json
import time

import anthropic

from .config import BATCH_STATE_FILE, MARKDOWN_PROMPT, MAX_BATCH_BYTES, MODEL, POLL_INTERVAL
from .textbook import TextbookSection, load_sections


def _build_request(section: TextbookSection) -> dict:
    # Builds one batch request for a section: all its page images followed by the extraction prompt
    # The Batch API will process this exactly like a normal messages.create() call
    content: list[dict] = []
    for path in section.image_paths:
        data = base64.standard_b64encode(path.read_bytes()).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data},
        })
    content.append({"type": "text", "text": MARKDOWN_PROMPT})
    return {
        "custom_id": section.custom_id,
        "params": {
            "model": MODEL,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": content}],
        },
    }


def _chunk_requests(requests: list[dict]) -> list[list[dict]]:
    # Splits requests into groups that stay under MAX_BATCH_BYTES of serialized JSON,
    # since the Batch API rejects a single submission over 256MB (mostly base64 image data)
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0
    for req in requests:
        size = len(json.dumps(req))
        if current and current_size + size > MAX_BATCH_BYTES:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(req)
        current_size += size
    if current:
        chunks.append(current)
    return chunks


def _write_section_md(section: TextbookSection, text: str) -> None:
    # Writes the extracted markdown to disk, prepending a YAML frontmatter block
    # The frontmatter lets the RAG pipeline read section metadata without parsing filenames
    frontmatter = (
        f"---\n"
        f"textbook: {section.textbook_id}\n"
        f"form: {section.form}\n"
        f"bab: {section.bab}\n"
        f'chapter_title: "{section.chapter_title}"\n'
        f'section: "{section.section_id}"\n'
        f'section_title: "{section.title}"\n'
        f"page_start: {section.page_start}\n"
        f"page_end: {section.page_end}\n"
        f"---\n\n"
    )
    section.output_path.parent.mkdir(parents=True, exist_ok=True)
    section.output_path.write_text(frontmatter + text, encoding="utf-8")


def _poll_and_save(
    client: anthropic.Anthropic,
    batch_ids: list[str],
    sections_by_id: dict[str, TextbookSection],
) -> tuple[int, int]:
    # Keeps asking the API "are you done yet?" every POLL_INTERVAL seconds until all batches finish.
    # Batches run concurrently server-side, so we just poll every id each round rather than
    # waiting for them one at a time.
    print("Polling for completion...")
    while True:
        totals = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0}
        all_ended = True
        for batch_id in batch_ids:
            batch = client.beta.messages.batches.retrieve(batch_id)
            c = batch.request_counts
            totals["processing"] += c.processing
            totals["succeeded"] += c.succeeded
            totals["errored"] += c.errored
            totals["canceled"] += c.canceled
            if batch.processing_status != "ended":
                all_ended = False
        print(
            f"  processing={totals['processing']}  succeeded={totals['succeeded']}"
            f"  errored={totals['errored']}  canceled={totals['canceled']}",
            end="\r",
        )
        if all_ended:
            break
        time.sleep(POLL_INTERVAL)
    print()

    # Stream results one by one and write each successful section to a .md file
    print("Saving results...")
    successes = 0
    failures = 0
    for batch_id in batch_ids:
        for result in client.beta.messages.batches.results(batch_id):
            section = sections_by_id.get(result.custom_id)
            if section is None:
                continue
            if result.result.type == "succeeded":
                # Join all text blocks from Claude's response into one string
                text = "".join(
                    block.text
                    for block in result.result.message.content
                    if hasattr(block, "text")
                )
                _write_section_md(section, text)
                successes += 1
            else:
                print(f"\n  ERROR [{result.custom_id}]: {result.result.error}")
                failures += 1

    return successes, failures


def run_extraction(sections: list[TextbookSection], force: bool = False) -> None:
    # Main entry point — decides whether to resume an existing batch or start a fresh one,
    # then drives the whole encode → submit → poll → save flow
    client = anthropic.Anthropic()

    # If batches were already submitted in a previous run, pick up where we left off
    if BATCH_STATE_FILE.exists() and not force:
        state = json.loads(BATCH_STATE_FILE.read_text())
        if state.get("status") == "in_progress":
            batch_ids = state["batch_ids"]
            print(f"Resuming {len(batch_ids)} pending batch(es) (submitted {state['submitted_at']})")
            custom_ids = set(state["custom_ids"])
            all_sections = load_sections()
            sections_by_id = {s.custom_id: s for s in all_sections if s.custom_id in custom_ids}
            successes, failures = _poll_and_save(client, batch_ids, sections_by_id)
            _finish(successes, failures)
            return

    # --force: wipe the old state file and start fresh
    if force and BATCH_STATE_FILE.exists():
        BATCH_STATE_FILE.unlink()

    # Skip sections that already have an output .md file (idempotency)
    pending = [s for s in sections if force or not s.is_extracted()]
    if not pending:
        print("All sections already extracted. Use --force to re-extract.")
        return

    # Encode every image in every pending section (this is the slow step)
    print(f"Encoding images and building {len(pending)} batch requests...")
    requests = []
    for i, section in enumerate(pending, 1):
        n_pages = len(section.image_paths)
        print(f"  [{i}/{len(pending)}] {section.custom_id} ({n_pages} pages)", end="\r", flush=True)
        requests.append(_build_request(section))
    print()

    # Split into multiple batches if the payload is too large for a single submission,
    # then send each chunk to the Batch API and save the batch IDs to disk
    chunks = _chunk_requests(requests)
    if len(chunks) > 1:
        print(f"Payload exceeds the batch size limit; splitting into {len(chunks)} batches...")

    print("Submitting batch(es)...")
    batch_ids = []
    submitted_at = None
    for i, chunk in enumerate(chunks, 1):
        batch = client.beta.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        submitted_at = submitted_at or batch.created_at.isoformat()
        print(f"  [{i}/{len(chunks)}] submitted: {batch.id} ({len(chunk)} requests)")

    BATCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BATCH_STATE_FILE.write_text(json.dumps({
        "batch_ids": batch_ids,
        "submitted_at": submitted_at,
        "custom_ids": [s.custom_id for s in pending],
        "status": "in_progress",
    }))

    sections_by_id = {s.custom_id: s for s in pending}
    successes, failures = _poll_and_save(client, batch_ids, sections_by_id)
    _finish(successes, failures)


def _finish(successes: int, failures: int) -> None:
    # Prints the final tally and cleans up the state file if everything went well
    print(f"\nDone. {successes} extracted, {failures} failed.")
    if failures == 0:
        BATCH_STATE_FILE.unlink(missing_ok=True)
    else:
        # Keep the state file so you can inspect which sections failed
        state = json.loads(BATCH_STATE_FILE.read_text())
        state["status"] = "ended_with_errors"
        BATCH_STATE_FILE.write_text(json.dumps(state))
        print(f"State file kept at {BATCH_STATE_FILE} for debugging.")
