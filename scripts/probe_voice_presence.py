"""生成 mp4 の音声に「人の声らしさ」がどれだけ含まれるかを客観指標で測る。

指標: 音声帯域(300-3400Hz)のエンベロープが、音節レート(4-8Hz)でどれだけ
変調されているか。人の声はこの帯域に強い変調を持ち、風・水・環境音は持たない。
speech_score = (4-8Hz の変調エネルギー) / (0.5-20Hz 全体の変調エネルギー)
"""
import sys, numpy as np, av
from scipy.signal import butter, sosfiltfilt, welch

def load_audio(path):
    c = av.open(path)
    if not c.streams.audio: return None, None
    st = c.streams.audio[0]; sr = st.rate
    chunks = [f.to_ndarray().astype(np.float32) for f in c.decode(audio=0)]
    c.close()
    a = np.concatenate([x.reshape(-1) if x.ndim == 1 else x.mean(axis=0) for x in chunks])
    if a.max() > 1.5: a = a / 32768.0
    return a, sr

def speech_score(a, sr):
    sos = butter(4, [300, min(3400, sr/2-1)], btype='band', fs=sr, output='sos')
    band = sosfiltfilt(sos, a)
    env = np.abs(band)
    # エンベロープを 200Hz へダウンサンプルしてから変調スペクトル
    ds = max(1, sr // 200); env = env[:len(env)//ds*ds].reshape(-1, ds).mean(axis=1)
    fs_env = sr / ds
    f, P = welch(env - env.mean(), fs=fs_env, nperseg=min(len(env), 1024))
    syl = P[(f >= 4) & (f <= 8)].sum()
    tot = P[(f >= 0.5) & (f <= 20)].sum()
    return float(syl / tot) if tot > 0 else 0.0

for path in sys.argv[1:]:
    a, sr = load_audio(path)
    if a is None:
        print(f"{path.split('/')[-1]:34} 音声なし"); continue
    print(f"{path.split('/')[-1]:34} rms={np.sqrt((a**2).mean()):.4f} speech_score={speech_score(a, sr):.3f}")
