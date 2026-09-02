#!/usr/bin/env python3
"""Local DEV-only nod event annotator.

Times are relative to each 60 s gold clip. TEST is never loaded.
Does not train, score, or write window labels.

  python scripts/annotate_nod_events_dev.py
  python scripts/annotate_nod_events_dev.py --port 8765
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.windowed_protocol import (  # noqa: E402
    CLIPS_DIR,
    NOD_DEFINITION,
    WindowedProtocolError,
    clip_records,
    dump_state,
    ensure_annotation_files,
    load_events,
    refuse_test_id,
    replace_clip_events,
    set_reviewed,
)

HTML = Path(__file__).with_suffix(".html")


def _event_records(df) -> list[dict]:
    if df is None or getattr(df, "empty", True):
        return []
    out = []
    for rec in df.to_dict(orient="records"):
        out.append(
            {
                "event_id": str(rec["event_id"]),
                "start_sec": float(rec["start_sec"]),
                "end_sec": float(rec["end_sec"]),
                "start_frame_relative": int(rec["start_frame_relative"]),
                "end_frame_relative": int(rec["end_frame_relative"]),
            }
        )
    return out


def _status_record(st) -> dict:
    row = st.iloc[0]
    return {
        "sample_id": str(row["sample_id"]),
        "reviewed": bool(row["reviewed"]),
        "n_nod_events": int(row["n_nod_events"]),
        "notes": str(row["notes"]),
    }


def clip_sec_for(sample_id: str) -> float:
    for row in clip_records():
        if row["sample_id"] == sample_id:
            return float(row["clip_sec"])
    raise WindowedProtocolError(f"unknown DEV sample {sample_id}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_mp4(self, path: Path) -> None:
        size = path.stat().st_size
        rng = self.headers.get("Range")
        if not rng:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(256 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            return
        unit, _, spec = rng.partition("=")
        if unit.strip() != "bytes":
            self.send_error(416)
            return
        start_s, _, end_s = spec.partition("-")
        try:
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
        except ValueError:
            self.send_error(416)
            return
        end = min(end, size - 1)
        if start < 0 or start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _json(self, code: int, obj: object) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self._send(code, raw, "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._json(200, dump_state())
            return
        if path.startswith("/video/"):
            sid = path.rsplit("/", 1)[-1]
            try:
                refuse_test_id(sid)
                clip_sec_for(sid)
            except WindowedProtocolError as e:
                self._json(404, {"error": str(e)})
                return
            mp4 = CLIPS_DIR / f"{sid}.mp4"
            if not mp4.is_file():
                self._json(404, {"error": f"no local clip for {sid}"})
                return
            self._send_mp4(mp4)
            return
        self._json(404, {"error": "not found"})

    def do_PUT(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        parts = [p for p in path.split("/") if p]
        try:
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "clip" and parts[3] == "events":
                sid = parts[2]
                payload = self._read_json()
                events = payload.get("events") or []
                rows, st = replace_clip_events(
                    sid,
                    events,
                    clip_sec=clip_sec_for(sid),
                    notes=payload.get("notes"),
                    reviewed=payload.get("reviewed"),
                )
                self._json(
                    200,
                    {
                        "events": _event_records(rows),
                        "status": _status_record(st),
                        "n_reviewed": int(dump_state()["n_reviewed"]),
                    },
                )
                return
            if len(parts) == 4 and parts[0] == "api" and parts[1] == "clip" and parts[3] == "status":
                sid = parts[2]
                payload = self._read_json()
                st = set_reviewed(
                    sid,
                    bool(payload.get("reviewed")),
                    notes=payload.get("notes"),
                    allow_zero_events=True,
                )
                ev = load_events()
                sub = ev[ev["sample_id"] == sid] if not ev.empty else ev
                self._json(
                    200,
                    {
                        "events": _event_records(sub),
                        "status": _status_record(st),
                        "n_reviewed": int(dump_state()["n_reviewed"]),
                    },
                )
                return
        except WindowedProtocolError as e:
            self._json(400, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001 — surface to the annotator UI
            self._json(500, {"error": str(e)})
            return
        self._json(404, {"error": "not found"})


def main() -> None:
    p = argparse.ArgumentParser(description="DEV nod event annotator (no TEST, no training).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    if not HTML.is_file():
        raise SystemExit(f"STOP: missing {HTML}")
    ensure_annotation_files()
    clips = clip_records()
    print(f"DEV clips loaded : {len(clips)}")
    print("TEST             : not loaded")
    print(f"Nod definition   : {NOD_DEFINITION[:80]}...")
    print(f"Open             : http://{args.host}:{args.port}/")
    print("Original gold CSVs are not written by this server.")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
