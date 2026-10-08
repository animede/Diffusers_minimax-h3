"""
ref2va の重ね投入 cadence プローブ(2026-10-07、単機32GB / 5090+8GB 検証で使用)。

2ワーカーで次のリクエストを即座に突っ込み(backend の「別の生成が進行中」は 0.3s
リトライで吸収)、H3_DECODE_STREAM の「decode 中に次の denoise が走る」重なりを
成立させた状態で完了間隔 = cadence を測る。直列 curl では decode が隠れず
total が 2〜5s 悪く見えるので、リアルタイム判定には必ずこちらを使うこと。

使い方(バックエンドは 8641 で起動済みであること):
    python3 ref2va_cadence_probe.py <width> <height> <本数>
例: python3 ref2va_cadence_probe.py 384 640 5

参照画像・音声のパスは下の D / AUD を環境に合わせて書き換える。
詳細な検証記録は docs/h3-single-gpu-32gb-20261007.md 参照。
"""
import sys, time, json, subprocess, threading
W, H, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
D = "/home/animede/minimax-h3/outputs/ab_ref2va"
AUD = "/tmp/claude-1000/probe7s_audio.wav"
done, lock = [], threading.Lock()
def gen(seed):
    out = f"/tmp/claude-1000/cad_{W}x{H}_{seed}.json"
    while True:
        subprocess.run(["curl","-s","-X","POST","http://127.0.0.1:8641/api/ref2va",
            "-F","prompt=The person in the reference photo sings passionately toward the camera with natural head movement and an expressive face, soft studio lighting, upper-body shot.",
            "-F",f"references=@{D}/input_image_person.png;type=image/png","-F",f"references=@{AUD};type=audio/wav",
            "-F",f"height={H}","-F",f"width={W}","-F","seconds=7.29","-F",f"seed={seed}",
            "-F","vocal_lock=true","-F","reference_image_short_edge=768","-o",out], check=True)
        r = json.load(open(out))
        if "detail" in r: time.sleep(0.3); continue
        with lock: done.append((time.time(), seed, r["total_elapsed_s"], r["peak_vram_gb"]))
        return
# 2ワーカーで重ね投入(前の decode 中に次の denoise を走らせる)
threads, seeds = [], list(range(9500, 9500+N))
for i, sd in enumerate(seeds):
    t = threading.Thread(target=gen, args=(sd,)); t.start(); threads.append(t)
    time.sleep(3 if i == 0 else 0.5)  # 先頭だけ少し先行させ、以後は即突っ込んで409リトライに任せる
for t in threads: t.join()
done.sort()
ts = [d[0] for d in done]
gaps = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
print(f"{W}x{H}: 完了間隔 = " + ", ".join(f"{g:.2f}" for g in gaps))
print(f"cadence(後半平均) = {sum(gaps[1:])/len(gaps[1:]):.2f}s / クリップ 7.29s, peakGPU0 = {max(d[3] for d in done):.2f}GB")
