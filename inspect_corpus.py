from music21 import corpus

files = list(corpus.getPaths())
midi_files = [path for path in files if str(path).lower().endswith(('.mid', '.midi'))]
print(f"total_paths={len(files)}")
print(f"midi_files={len(midi_files)}")
for path in midi_files[:30]:
    print(path)
