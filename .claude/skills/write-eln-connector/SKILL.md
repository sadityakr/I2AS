---
name: write-eln-connector
description: Write an I2AS ELN connector block — the small class that lets I2AS link, read and publish to an electronic lab notebook other than eLabFTW (or a customised eLabFTW). Use when the user wants I2AS to talk to their notebook (RSpace, LabArchives, Benchling, SciNote, openBIS, a lab wiki with an API, …). Loops on `python -m i2as.blocks check` until every rule passes.
---

# Write an ELN connector block

A connector is a **tier-2 user block**: one Python file, one class, that
talks to ONE notebook's API. I2AS does everything else — queuing, retries,
idempotency, rendering, approval, the API key — and runs the connector in a
helper process, so a bug in it costs a visible error and a retry, never the
experiment. Keep it small and synchronous.

## Steps

1. **Read the contract** at the top of `i2as/blocks/connector.py` and the
   reference connector `i2as/blocks/shipped/connectors/elabftw.py`. The public
   API is EXACTLY: `verify`, `list_templates`, `search`, `get_record`,
   `create_entry`, `append_section`, `has_section`, `set_fields`, `upload`,
   `link_item`. No other public method.
2. **Find the notebook's API**: authentication header, "who am I", search,
   read a page and an item (its structured fields), create a page (from a
   template if supported), update a page's body, upload a file, link an item.
   Ask the user for the API docs URL if you cannot find them.
3. **Scaffold** into the user's blocks folder:
   `python -m i2as.blocks new-connector <backend_id>` (writes
   `<user config>/blocks/connectors/<backend_id>.py`; `--dir` to choose).
4. **Implement** each method:
   - All HTTP goes through `self._transport.request(...)` (an
     `ElnHttpTransport`), then `raise_for_status(method, path, response)` — it
     turns 401/403 into `ElnAuthError`, 404 into `ElnNotFound`, other 4xx into
     `ElnValidationError`, 429/5xx into `ElnTransientError`.
   - Raise ONLY `ElnError` subclasses. A feature the notebook lacks raises
     `ElnValidationError("… not supported")` and its capability flag is False.
   - `append_section(entry, publish_id, html)` must ADD the HTML at the END of
     the page and never change what is there (read the body, append, write
     back if the API has no append). `has_section` answers whether
     `publish_id` (plain text in the section heading) is already on the page.
   - `set_fields` overwrites the named structured fields; `get_record` returns
     fields flattened to `{name: text}` (and `units`).
   - Put the API key only in the auth header — never in a message or a log.
5. **Declare** `backend` (lowercase id), `display_name`, `capabilities`
   (`ElnCapabilities(...)` with literal flags) and `settings_schema` (a
   LITERAL dict: `{"properties": {name: {"type": "string|boolean|number|integer",
   "title", "description", "default"}}, "required": [...]}` — the Settings
   dialog renders its form from it; no secret in it: the key is the
   `credential` argument).
6. **Check, and loop until it passes**:
   `python -m i2as.blocks check <file>`
   Fix every `FAIL` line and run it again. Do not stop while any line fails.
7. **Try it against the real notebook** (with the user): Settings →
   Electronic notebook → choose the connector, fill the form, paste the key →
   "Save & test connection". Then link an experiment's page and publish once.
8. Tell the user where the file is and that editing it later requires
   confirming the new version for experiments already linked.

## Rules

- Standard library and `i2as.blocks` only (no `requests`): use
  `UrllibTransport` / `multipart_file` from `i2as.blocks`.
- No module-level side effects; no global state; no threads.
- Never print the credential; never write files outside what `upload` is given.
