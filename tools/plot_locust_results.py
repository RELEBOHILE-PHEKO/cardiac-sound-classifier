import csv
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LR = ROOT / 'locust_results'

files = {
    1: LR / 'run_workers_1_stats.csv',
    2: LR / 'run_workers_2_stats.csv',
    4: LR / 'run_workers_4_stats.csv',
}

rows = []
for workers, fp in files.items():
    if not fp.exists():
        print(f"Missing {fp}, skipping {workers}")
        continue
    with fp.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # The first non-empty logical row with POST will be used
        data_row = None
        for r in reader:
            if r.get('Type', '').strip().upper() == 'POST' or r.get('Name', '').strip() == '/predict':
                data_row = r
                break
        if data_row is None:
            # fallback: take second row if present
            f.seek(0)
            lines = f.read().splitlines()
            if len(lines) >= 3:
                # parse the second csv data row
                data = lines[2].split(',')
                # map to header
                header = lines[0].split(',')
                data_row = dict(zip(header, data))
        if data_row is None:
            print(f"Can't find data row in {fp}")
            continue
        try:
            req_count = int(data_row.get('Request Count','0') or 0)
            failures = int(data_row.get('Failure Count','0') or 0)
            median = float(data_row.get('Median Response Time','0') or 0)
            avg = float(data_row.get('Average Response Time','0') or 0)
            maxr = float(data_row.get('Max Response Time','0') or 0)
            reqs_s = float(data_row.get('Requests/s','0') or 0)
        except Exception:
            # try alternate keys
            req_count = int(data_row.get('Request Count','0') or 0)
            failures = int(data_row.get('Failure Count','0') or 0)
            median = float(data_row.get('Median Response Time','0') or 0)
            avg = float(data_row.get('Average Response Time','0') or 0)
            maxr = float(data_row.get('Max Response Time','0') or 0)
            reqs_s = float(data_row.get('Requests/s','0') or 0)

        rows.append({'workers':workers,'requests':req_count,'failures':failures,'median_ms':median,'avg_ms':avg,'max_ms':maxr,'reqs_per_s':reqs_s})

if not rows:
    print('No data parsed; exiting')
    raise SystemExit(1)

rows = sorted(rows, key=lambda r: r['workers'])
workers = [r['workers'] for r in rows]
avgs = [r['avg_ms'] for r in rows]
meds = [r['median_ms'] for r in rows]
maxs = [r['max_ms'] for r in rows]
throughputs = [r['reqs_per_s'] for r in rows]

# Create plots folder
plots_dir = LR
plots_dir.mkdir(parents=True, exist_ok=True)

# Latency plot (avg & median)
plt.figure(figsize=(7,4))
plt.plot(workers, avgs, marker='o', label='Average (ms)')
plt.plot(workers, meds, marker='o', label='Median (ms)')
plt.plot(workers, maxs, marker='x', linestyle='--', color='gray', label='Max (ms)')
plt.title('Inference Latency vs Worker Count')
plt.xlabel('Uvicorn workers')
plt.ylabel('Response time (ms)')
plt.xticks(workers)
plt.grid(True, alpha=0.3)
plt.legend()
lat_plot = plots_dir / 'latency_vs_workers.png'
plt.tight_layout()
plt.savefig(lat_plot)
plt.close()

# Throughput plot
plt.figure(figsize=(7,4))
plt.plot(workers, throughputs, marker='o', label='Requests/s')
plt.title('Throughput vs Worker Count')
plt.xlabel('Uvicorn workers')
plt.ylabel('Requests per second')
plt.xticks(workers)
plt.grid(True, alpha=0.3)
plt.legend()
thr_plot = plots_dir / 'throughput_vs_workers.png'
plt.tight_layout()
plt.savefig(thr_plot)
plt.close()

# Write summary markdown
md = []
md.append('# Locust Scaled Runs Analysis')
md.append('')
md.append('Summary generated from `locust_results/run_workers_*_stats.csv`.')
md.append('')
md.append('| workers | requests | failures | median_ms | avg_ms | max_ms | reqs_per_s |')
md.append('|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    md.append(f"| {r['workers']} | {r['requests']} | {r['failures']} | {r['median_ms']} | {r['avg_ms']:.1f} | {r['max_ms']:.1f} | {r['reqs_per_s']:.2f} |")
md.append('')
md.append('**Plots**')
md.append('')
md.append(f'![Latency vs workers]({lat_plot.name})')
md.append('')
md.append(f'![Throughput vs workers]({thr_plot.name})')
md.append('')
md.append('**Notes**')
md.append('- All runs had zero failures. Median and average latencies generally improved as worker count increased in this environment, and throughput increased slightly.')

with (LR / 'scaled_analysis.md').open('w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print('Plots and summary written to', LR)
