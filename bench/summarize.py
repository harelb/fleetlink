"""Markdown tables from multi_robot_bench.py JSON lines.

    python bench/summarize.py results.jsonl [more.jsonl ...]
"""

import json
import sys


def fmt(v, nd=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def rows(paths):
    for p in paths:
        for line in open(p):
            if line.strip():
                yield json.loads(line)


def main(paths):
    runs = list(rows(paths))
    print("| robots | Mbit/s | video | mirror | variant | air Mbit/s | busy | C1 age p99 / max s | cmd RTT p99 s | "
          "graph catch-up max / after LC s | mesh after reveal / after LC s (unfinished) | video freezes, min fps | "
          "mirror backlog, max delay s |")  # fmt: skip
    print("|" + "---|" * 13)
    for r in runs:
        s = r["summary"]
        var = []
        if r.get("video_preset") and r["video"]:
            var.append(r["video_preset"])
        if r.get("hops") and max(r["hops"]) > 1:
            var.append(f"last robot {max(r['hops']):g} hops")
        if r.get("tcp_hol"):
            var.append("TCP HOL")
        unfinished = s["mesh_lc_unfinished"]  # robots whose mesh never converged after the LC
        late = s["mesh_reveal_unfinished"]  # not yet converged when the LC hit
        mesh_cell = fmt(s["mesh_after_reveal"], 0) + (f" [{late} late]" if late else "")
        mesh_cell += " / " + fmt(s["mesh_after_lc"], 0) + (f" ({unfinished} never)" if unfinished else "")
        print(
            f"| {s['robots']} | {s['rate_mbps']:g} | {s['video']} | {'on' if s['mirror'] else 'off'} | {', '.join(var)} "
            f"| {fmt(s['air_mbps'], 2)} | {fmt(s['channel_busy'], 2)} "
            f"| {fmt(s['state_age_p99'], 2)} / {fmt(s['state_age_max'], 2)} | {fmt(s['cmd_rtt_p99'], 2)} "
            f"| {fmt(s['c2_catchup_max'], 0)} / {fmt(s['c2_after_lc'], 0)} "
            f"| {mesh_cell} "
            f"| {fmt(s['video_freezes'])}, {fmt(s['video_fps_min'])} "
            f"| {fmt(s['mirror_backlog'])}, {fmt(s['mirror_delay_max'], 0)} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
