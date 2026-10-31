"""Cut a time-compressed demo video from screen captures and the agent's MCP transcript.

Inputs are described by an edit file (JSON):

    {
      "pieces": [ {"file": "raw.mkv", "epoch": <capture start, unix s>,
                   "spans": [[start_s, end_s, speed], ...]},
                  {"card": "Text shown full-screen", "seconds": 2.0}, ... ],
      "transcript": "transcript.jsonl",
      "headlines": [[headline, detail], ...],   # one per narration, in order
      "skip_tools": ["read_readings", ...],     # calls not shown in the overlay
      "title": [...lines...], "end": [...lines...],
      "size": [2560, 1440], "band_top": 1312
    }

Each capture span is played back at its own speed; the overlay (in the band
below the application windows) shows the agent's narration as a headline —
held at least ``MIN_HOLD`` seconds of video so it can be read, queued if the
agent spoke faster than that — the last MCP tool calls with their verdicts,
and the real elapsed time with the current speed-up. Output: an MP4 at the
capture resolution and a 1080p copy.

Usage: python scripts/agent_demo/make_video.py <edit.json> <out_dir>
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FPS = 30
MIN_HOLD = 2.6
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def card_png(path: Path, lines: list[str], size: tuple[int, int]) -> None:
    """Render a dark card: first line large, the rest smaller."""
    w, h = size
    image = Image.new("RGB", size, (22, 26, 33))
    draw = ImageDraw.Draw(image)
    big = ImageFont.truetype(FONT_BOLD, int(h * 0.05))
    small = ImageFont.truetype(FONT, int(h * 0.026))
    y = int(h * 0.30)
    for index, line in enumerate(lines):
        font = big if index == 0 else small
        width = draw.textlength(line, font=font)
        draw.text(((w - width) / 2, y), line, font=font, fill=(235, 238, 242) if index == 0 else (170, 182, 196))
        y += int(font.size * (1.9 if index == 0 else 1.55))
    image.save(path)


def main() -> int:
    edit_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    edit = json.loads(edit_path.read_text())
    base = edit_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = edit.get("size", [2560, 1440])
    band_top = edit.get("band_top", 1312)

    # ── 1. Cut every piece to a 30 fps clip, building the time map as we go ──
    clips: list[Path] = []
    time_map: list[tuple[float, float, float, float]] = []  # (epoch_start, epoch_end, out_start, speed)
    out_t = 0.0
    pieces = [{"card": edit["title"], "seconds": edit.get("title_seconds", 3.0)}]
    pieces += edit["pieces"]
    pieces += [{"card": edit["end"], "seconds": edit.get("end_seconds", 7.0)}]
    for index, piece in enumerate(pieces):
        clip = out_dir / f"clip_{index:03d}.mp4"
        if "card" in piece:
            lines = piece["card"] if isinstance(piece["card"], list) else [piece["card"]]
            png = out_dir / f"card_{index:03d}.png"
            card_png(png, lines, (width, height))
            run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-i", str(png), "-t", str(piece["seconds"]),
                 "-r", str(FPS), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(clip)])
            clips.append(clip)
            out_t += float(piece["seconds"])
            continue
        source = base / piece["file"]
        for span_index, (start, end, speed) in enumerate(piece["spans"]):
            span_clip = out_dir / f"clip_{index:03d}_{span_index:02d}.mp4"
            run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-to", str(end), "-i", str(source),
                 "-vf", f"setpts=(PTS-STARTPTS)/{speed},fps={FPS}", "-an", "-c:v", "libx264", "-crf", "18",
                 "-pix_fmt", "yuv420p", str(span_clip)])
            clips.append(span_clip)
            time_map.append((piece["epoch"] + start, piece["epoch"] + end, out_t, float(speed)))
            out_t += (end - start) / speed
    total = out_t

    def to_out(epoch: float) -> float | None:
        for e0, e1, o0, speed in time_map:
            if e0 <= epoch <= e1:
                return o0 + (epoch - e0) / speed
        return None

    # ── 2. The overlay, in the output timeline ──
    records = [json.loads(line) for line in (base / edit["transcript"]).read_text().splitlines() if line.strip()]
    skip = set(edit.get("skip_tools", []))
    first_epoch = time_map[0][0]
    events: list[str] = []

    narrations = [r for r in records if r["kind"] == "narration"]
    shown_until = 0.0
    starts: list[tuple[float, int]] = []
    for index, record in enumerate(narrations):
        t = to_out(record["t"])
        if t is None:
            continue
        start = max(t, shown_until)
        starts.append((start, index))
        shown_until = start + MIN_HOLD
    end_of_body = total - edit.get("end_seconds", 7.0)
    for position, (start, index) in enumerate(starts):
        stop = starts[position + 1][0] if position + 1 < len(starts) else end_of_body
        headline, detail = edit["headlines"][index]
        text = "{\\c&HFFE066&\\b1}AGENT  {\\c&HFFFFFF&}" + ass_escape(headline)
        if detail:
            text += "\\N{\\fs24\\b0\\c&HC8C8C8&}" + ass_escape(detail)
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(stop)},Narr,,0,0,0,,{text}")

    calls = [r for r in records if r["kind"] == "result" and (r.get("params") or {}).get("name") not in skip]
    shown: list[tuple[float, str]] = []
    for record in calls:
        t = to_out(record["t"])
        if t is None:
            continue
        name = record["params"].get("name") or record["method"]
        answer = record.get("answer") or {}
        try:
            content = json.loads(answer["result"]["content"][0]["text"])
            code = content.get("code", "")
        except (KeyError, TypeError, ValueError, IndexError):
            code = "ERROR"
        mark = "{\\c&H7CE07C&}✓" if code in ("OK", "PENDING") else "{\\c&H6060FF&}✗"
        shown.append((t, f"{mark}{{\\c&HE6E6E6&}} {name}"))
    for position, (t, _) in enumerate(shown):
        stop = shown[position + 1][0] if position + 1 < len(shown) else end_of_body
        window = [label for _, label in shown[max(0, position - 3): position + 1]]
        text = "{\\c&H999999&\\fs22}MCP tool calls\\N" + "\\N".join(window)
        events.append(f"Dialogue: 1,{ass_time(t)},{ass_time(max(stop, t + 0.05))},Calls,,0,0,0,,{text}")

    # The real-time timer: elapsed real time since the capture started (large),
    # the wall-clock time and the current speed-up beneath it — one event per
    # video frame, so it visibly races through the compressed stretches.
    step = 1.0 / FPS
    t = time_map[0][2]
    while t < end_of_body:
        segment = next(((e0, o0, speed) for e0, e1, o0, speed in time_map if o0 <= t < o0 + (e1 - e0) / speed), None)
        if segment is not None:
            e0, o0, speed = segment
            epoch = e0 + (t - o0) * speed
            h, rem = divmod(int(epoch - first_epoch), 3600)
            m, s = divmod(rem, 60)
            wall = time.strftime("%H:%M:%S", time.gmtime(epoch))
            text = (
                "{\\fs20\\c&H999999&}REAL TIME\\N"
                f"{{\\fs60\\b1\\c&HFFFFFF&}}{h:02d}:{m:02d}:{s:02d}\\N"
                f"{{\\fs22\\b0\\c&HB0B0B0&}}{wall} UTC   {{\\c&HFFE066&}}×{speed:g}"
            )
            events.append(f"Dialogue: 2,{ass_time(t)},{ass_time(t + step)},Timer,,0,0,0,,{text}")
        t += step

    band_mid = band_top + (height - band_top) // 2
    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Narr,DejaVu Sans,31,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,4,36,1100,0,1
Style: Calls,DejaVu Sans Mono,24,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,4,1830,30,0,1
Style: Timer,DejaVu Sans Mono,24,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Positions: the narration and calls are vertically centred in the band.
    events = [
        e.replace(",Narr,,0,0,0,,", f",Narr,,0,0,0,,{{\\pos(36,{band_mid})}}")
        .replace(",Calls,,0,0,0,,", f",Calls,,0,0,0,,{{\\pos(1840,{band_mid})}}")
        .replace(",Timer,,0,0,0,,", f",Timer,,0,0,0,,{{\\pos(1655,{band_mid})}}")
        for e in events
    ]
    (out_dir / "overlay.ass").write_text(ass + "\n".join(events) + "\n", encoding="utf-8")

    # ── 3. Concatenate, burn the overlay, export ──
    listing = out_dir / "clips.txt"
    listing.write_text("".join(f"file '{clip.name}'\n" for clip in clips))
    body = out_dir / "body.mp4"
    run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(body)])
    final = out_dir / "agent_demo_1440p.mp4"
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(body), "-vf", f"ass={out_dir / 'overlay.ass'}",
         "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)])
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(final), "-vf", "scale=1920:1080:flags=lanczos",
         "-c:v", "libx264", "-crf", "21", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(out_dir / "agent_demo_1080p.mp4")])
    print(f"{total:.1f} s -> {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
