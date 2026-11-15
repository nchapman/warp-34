"""Shared plan/run selection and explicit non-measurement coverage records."""


def availability(engine):
    support = engine.get("support", "unverified")
    if support == "unsupported":
        return "unsupported", engine["skip_reason"]
    if not engine.get("enabled", True):
        return "disabled", engine["skip_reason"]
    return "ready", None


def select(cfg, engine_ids=None, case_ids=None, *, gpu_count=None, exclude_ids=None):
    for selection, available, label in ((engine_ids, cfg["engines"], "engine"),
                                        (exclude_ids or None, cfg["engines"], "excluded engine"),
                                        (case_ids, cfg["cases"], "case")):
        if selection is not None and (not selection or set(selection) - {item["id"] for item in available}):
            raise ValueError(f"unknown or empty {label} selection")
    if gpu_count is not None and (isinstance(gpu_count, bool) or gpu_count not in (1, 2)):
        raise ValueError("gpu_count filter must be 1 or 2")
    cases = [c for c in cfg["cases"] if case_ids is None or c["id"] in case_ids]
    engines, coverage = [], []
    for engine in cfg["engines"]:
        status, reason = availability(engine)
        # Preserve genuine unsupported/disabled reasons even when also filtered out.
        if status == "ready":
            if engine_ids is not None and engine["id"] not in engine_ids:
                status, reason = "not_selected", "not requested by engine selection"
            elif engine["id"] in (exclude_ids or []):
                status, reason = "excluded", "excluded by operator"
            elif gpu_count is not None and engine["gpu_count"] != gpu_count:
                status, reason = "filtered", f"campaign restricted to {gpu_count} GPU(s)"
        if status == "ready":
            engines.append(engine)
        for case in cases:
            coverage.append({"engine_id": engine["id"], "case_id": case["id"],
                             "gpu_count": engine["gpu_count"],
                             "support": engine.get("support", "unverified"),
                             "status": "not_run" if status == "ready" else status,
                             "reason": reason, "completed_repetitions": 0,
                             "required_repetitions": cfg["run"]["repetitions"]})
    return engines, cases, coverage
