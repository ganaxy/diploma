import sys, time, statistics
sys.path.insert(0, r'C:\Users\M Tech\Desktop\diplom')

from gradio_app.models import FlatClassifier, TwoStageClassifier, pick_device

device, device_label = pick_device()
print(f'Device: {device_label}')

flat = FlatClassifier(device)
two  = TwoStageClassifier(device)

texts = [
    'Ийм эргүү тэнэг хүмүүс байх ч гэж. Хэлэх муу үг алга.',
    'Энэ шинэчлэл бүрэн дутуу, хариуцлагатай хүмүүс төлөвлөгөөгөө дахин хянах хэрэгтэй.',
    'Сайн байна уу. Та 11-11 дугаарт холбогдон мэдээлэл авах боломжтой.',
    'Ёстой их зориг шүү бүтэн 4 цаг далайд сэлнэ гэдэг төсөөлшгүй.',
    'Засгийн газрын энэ шийдвэр бол улс орны хөгжилд саад болж байна, яаралтай өөрчлөх хэрэгтэй.',
]

N = 20
print(f'Benchmarking {N} runs x {len(texts)} texts...\n')

flat_times, two_times, two_branched = [], [], 0
for text in texts:
    for _ in range(N):
        t0 = time.perf_counter()
        flat.predict(text)
        flat_times.append((time.perf_counter()-t0)*1000)
    for _ in range(N):
        t0 = time.perf_counter()
        r = two.predict(text)
        two_times.append((time.perf_counter()-t0)*1000)
        if r.branched_to_stage2:
            two_branched += 1

flat_mean = statistics.mean(flat_times)
flat_p95  = sorted(flat_times)[int(0.95*len(flat_times))]
two_mean  = statistics.mean(two_times)
two_p95   = sorted(two_times)[int(0.95*len(two_times))]
reduction = (two_mean - flat_mean) / two_mean * 100
total_runs = N * len(texts)

print('=== FLAT MODEL ===')
print(f'  Mean : {flat_mean:.1f} ms')
print(f'  P95  : {flat_p95:.1f} ms')
print(f'  Min  : {min(flat_times):.1f} ms')
print(f'  Max  : {max(flat_times):.1f} ms')

print()
print('=== TWO-STAGE MODEL ===')
print(f'  Mean : {two_mean:.1f} ms')
print(f'  P95  : {two_p95:.1f} ms')
print(f'  Stage 2 triggered: {two_branched}/{total_runs} ({two_branched/total_runs*100:.0f}%)')

print()
print('=== RESULT ===')
print(f'  Latency reduction (Flat vs Two-Stage): {reduction:.1f}%')
pass_19 = 'PASS' if flat_mean <= 100 else 'FAIL'
pass_5  = 'PASS' if reduction >= 20 else 'PARTIAL'
print(f'  HO-19 (<=100ms real-time): {pass_19} — Flat mean = {flat_mean:.1f}ms')
print(f'  HO-5  (>=20% reduction):   {pass_5} — {reduction:.1f}%')
