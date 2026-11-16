"""Concise reports from independently reconstructed operating quantities."""


def render_markdown(payload):
    s = payload["summary"]
    lines = [
        "# Operating validation",
        "",
        f"Model: `{s['model_version']}`. Validation: {'passed' if s['checks']['passed'] else 'FAILED'}.",
        f"Flow policy: `{s['flow_policy']}` (prefix = proportions at each first leg; aggregate = task totals across first legs).",
        "",
        f"Total cost: {s['objective']:.6f} per hour.",
        f"Orders: {s['demand']:.6f}/h; drone {s['served']:.6f}/h; direct courier {s['direct']:.6f}/h.",
        f"Service activities: {s['single_tasks']:.6f} single + {s['double_tasks']:.6f} double per hour.",
        f"Continuous fleet: {s['fleet']:.6f}; installed compartments: {s['slot_total']:.0f}.",
        "",
        "Task rates count flights; order rates count each served order occurrence, including repeated ODs.",
        "Delivery and endurance limits apply to means. Locker overflow is the Poisson offered-load approximation.",
        "",
        "| Check | Residual |",
        "|---|---:|",
    ]
    lines += [
        f"| {key} | {value:.3g} |" for key, value in s["checks"]["residuals"].items()
    ]
    if s["checks"]["physical_errors"]:
        lines += ["", *s["checks"]["physical_errors"]]
    if payload.get("metric_notes"):
        lines += ["", "## Metric definitions and interpretation", ""]
        lines += ["- " + note for note in payload["metric_notes"]]
        lines += [
            "",
            "See `metric_dictionary.csv` for every table field and unit.",
            "",
            "## Detailed exported tables",
            "",
            "| Table | Rows |",
            "|---|---:|",
        ]
        lines += [
            f"| {name}.csv | {len(rows)} |" for name, rows in payload["tables"].items()
        ]
    return "\n".join(lines) + "\n"
