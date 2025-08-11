import os

# === CONFIGURATION ===
filelists_dir = r"C:\Users\yossi\Documents\Bar Ilan\Master's\תשפה - סמסטר א\מודלים גנרטיביים עמוקים\Final_project\NSTTS\Speech-Backbones\Grad-TTS\resources\filelists\ljspeech"
path_to_audio_dir = r"C:\Users\yossi\Documents\Bar Ilan\Master's\תשפה - סמסטר א\מודלים גנרטיביים עמוקים\Final_project\NSTTS\datasets\LJSpeech-1.1\LJSpeech-1.1\wavs"

# === PROCESS ALL TXT FILELISTS ===
for filename in os.listdir(filelists_dir):
    if filename.endswith(".txt"):
        full_path = os.path.join(filelists_dir, filename)
        output_path = os.path.join(filelists_dir, filename.replace(".txt", "_fixed.txt"))
        print(f"🔧 Processing: {filename}")

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            wav_file = os.path.basename(parts[0])  # e.g., LJ001-0001.wav
            fixed_path = os.path.join(path_to_audio_dir, wav_file)
            new_line = f"{fixed_path}|{parts[1]}"
            if len(parts) > 2:
                new_line += "|" + "|".join(parts[2:])
            fixed_lines.append(new_line + "\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(fixed_lines)

        print(f"✅ Saved fixed filelist to: {output_path}")
