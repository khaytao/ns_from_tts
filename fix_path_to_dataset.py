# Fix LJSpeech filelists (simple, interactive, robust)
from pathlib import Path
import os
import sys

def choose(prompt: str, options: list[Path]) -> Path:
    """Simple CLI picker. Returns the selected Path."""
    print(prompt)
    for i, p in enumerate(options, 1):
        print(f"  [{i}] {p}")
    while True:
        s = input(f"Choose 1-{len(options)}: ").strip()
        if s.isdigit():
            k = int(s)
            if 1 <= k <= len(options):
                return options[k - 1]
        print("Invalid choice, try again.")

def find_filelists_dir() -> Path:
    """Find candidate ljspeech filelist dirs and let the user pick one."""
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()

    # Seed candidates with the common locations
    candidates = [
        here / "resources" / "filelists" / "ljspeech",
        cwd  / "resources" / "filelists" / "ljspeech",
    ]

    # Also scan recursively under CWD for any .../resources/filelists/ljspeech
    for p in cwd.rglob("resources/filelists/ljspeech"):
        candidates.append(p)

    # Keep unique, existing dirs that have at least one .txt file
    uniq: list[Path] = []
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            continue
        if rp in seen:
            continue
        if rp.is_dir() and any(rp.glob("*.txt")):
            seen.add(rp)
            uniq.append(rp)

    if not uniq:
        # Nothing found: ask the user
        manual = input("Path to filelists dir (the folder with train.txt/test.txt/valid.txt): ").strip()
        if not manual:
            print("❌ No filelists dir provided.")
            sys.exit(1)
        d = Path(manual).resolve()
        if not d.is_dir():
            print(f"❌ Not a directory: {d}")
            sys.exit(1)
        return d

    if len(uniq) == 1:
        return uniq[0]

    # More than one candidate -> let the user choose
    return choose("Found multiple filelists directories:", uniq)

def get_audio_dir() -> Path:
    """Use AUDIO_DIR env var or prompt."""
    audio = os.environ.get("AUDIO_DIR", "").strip()
    if not audio:
        audio = input("Path to LJSpeech 'wavs' folder: ").strip()
    if not audio:
        print("❌ No audio_dir provided.")
        sys.exit(1)
    d = Path(audio).resolve()
    if not d.is_dir():
        print(f"❌ audio_dir not found: {d}")
        sys.exit(1)
    return d

def main():
    filelists_dir = find_filelists_dir()
    audio_dir = get_audio_dir()

    print(f"📄 Filelists dir: {filelists_dir}")
    print(f"🎧 Audio dir:     {audio_dir}")

    # Only process plain .txt (skip already fixed files)
    txt_files = sorted(p for p in filelists_dir.glob("*.txt") if not p.name.endswith("_fixed.txt"))
    if not txt_files:
        print("⚠️ No .txt filelists found to process.")
        return

    for src in txt_files:
        dst = src.with_name(src.stem + "_fixed.txt")
        print(f"🔧 {src.name} → {dst.name}")

        lines = src.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        out = []
        total = fixed = missing = 0

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue

            wav_file = Path(parts[0]).name
            wav_path = audio_dir / wav_file

            if wav_path.is_file():
                new_line = "|".join([str(wav_path)] + parts[1:])
                out.append(new_line + "\n")
                fixed += 1
            else:
                # keep original if missing
                out.append(raw if raw.endswith("\n") else raw + "\n")
                missing += 1

            total += 1

        dst.write_text("".join(out), encoding="utf-8")
        print(f"   done. lines={total}, fixed={fixed}, missing={missing} (kept as-is)")

    print("✅ All done.")

if __name__ == "__main__":
    main()
