import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    locust_dir = Path('outputs/locust')
    hist_file = locust_dir / 'run1_stats_history.csv'
    if not hist_file.exists():
        print(f'History file not found: {hist_file}')
        return 1

    df = pd.read_csv(hist_file)

    # Clean up columns (some exports have N/A entries)
    # Convert numeric columns where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            pass

    # Use a simple index for time (row number)
    x = range(len(df))

    # Plot Requests/s over time
    plt.figure(figsize=(10, 4))
    plt.plot(x, df['Requests/s'].fillna(0), label='Requests/s')
    plt.xlabel('Sample')
    plt.ylabel('Requests/s')
    plt.title('Locust - Requests per Second (run1)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = locust_dir / 'locust_requests.png'
    plt.savefig(out1, dpi=200)
    print('Saved', out1)

    # Plot median and average response time over time
    plt.figure(figsize=(10, 4))
    if 'Total Median Response Time' in df.columns:
        plt.plot(x, df['Total Median Response Time'].fillna(0), label='Median (Total)')
    if 'Total Average Response Time' in df.columns:
        plt.plot(x, df['Total Average Response Time'].fillna(0), label='Average (Total)')
    plt.xlabel('Sample')
    plt.ylabel('Response Time (ms)')
    plt.title('Locust - Response Times (run1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = locust_dir / 'locust_response_times.png'
    plt.savefig(out2, dpi=200)
    print('Saved', out2)

    # Summarize key metrics
    total_requests = int(df['Total Request Count'].dropna().iloc[-1]) if 'Total Request Count' in df.columns and not df['Total Request Count'].dropna().empty else 0
    total_failures = int(df['Total Failure Count'].dropna().iloc[-1]) if 'Total Failure Count' in df.columns and not df['Total Failure Count'].dropna().empty else 0
    peak_rps = float(df['Requests/s'].max()) if 'Requests/s' in df.columns else 0.0
    median_resp = float(df['Total Median Response Time'].dropna().iloc[-1]) if 'Total Median Response Time' in df.columns and not df['Total Median Response Time'].dropna().empty else 0.0
    avg_resp = float(df['Total Average Response Time'].dropna().iloc[-1]) if 'Total Average Response Time' in df.columns and not df['Total Average Response Time'].dropna().empty else 0.0
    max_resp = float(df['Total Max Response Time'].max()) if 'Total Max Response Time' in df.columns else 0.0

    summary = (
        f"Total requests: {total_requests}\n"
        f"Total failures: {total_failures}\n"
        f"Peak requests/s: {peak_rps:.1f}\n"
        f"Median response time (end): {median_resp:.1f} ms\n"
        f"Average response time (end): {avg_resp:.1f} ms\n"
        f"Max observed response time: {max_resp:.1f} ms\n"
    )

    summary_file = locust_dir / 'summary.txt'
    summary_file.write_text(summary)
    print('Wrote summary to', summary_file)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
