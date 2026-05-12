# Screenshots

Drop UI screenshots here. The README references:

- `list_view.png` -- the filterable, paginated table at `localhost:8501`
- `detail_view.png` -- a single trace open via `?trace_id=<id>`

These are intentionally not auto-generated; the sandbox running CI/Cursor agents has no browser, so screenshotting is a manual step. To refresh:

```bash
bash scripts/run_ui.sh
# then in a real browser: screenshot the list view + open one trace, screenshot.
```
