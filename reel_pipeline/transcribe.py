import json,sys
from faster_whisper import WhisperModel
m=WhisperModel(sys.argv[3],device="cpu",compute_type="int8")
import numpy as np,wave
wv=wave.open(sys.argv[1]);a=np.frombuffer(wv.readframes(wv.getnframes()),np.int16).astype(np.float32)/32768
segs,info=m.transcribe(a,word_timestamps=True,vad_filter=False)
out=[]
print(info.language,info.language_probability)
for s in segs:
    print(f"[{s.start:.2f}-{s.end:.2f}] {s.text}")
    for w in s.words: out.append([w.start,w.end,w.word.strip()])
json.dump(out,open(sys.argv[2],"w"),indent=0)
