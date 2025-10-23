import json
from collections import defaultdict

with open("/work/logs/profiler/20251005-064929/plugins/profile/de89b13a7b6e/de89b13a7b6e_2109944.1759647077789824613.pt.trace.json") as f:
    trace = json.load(f)

# FLOPSごとに集計
flops_cpu = defaultdict(float)
flops_cuda = defaultdict(float)
# どんなキーがあるか確認
for event in trace['traceEvents'][:100]:
    print(event.get('args', {}).keys())
for event in trace['traceEvents']:
    name = event.get('name')
    cat = event.get('cat', '')
    args = event.get('args', {})
    flops = args.get('Flops', 0)
    if flops:
        if 'cuda' in cat.lower():
            flops_cuda[name] += flops
        else:
            flops_cpu[name] += flops

def print_top_flops(flops_dict, label, n=10):
    top = sorted(flops_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    print(f"--- {label} FLOPSランキング 上位{n}件 ---")
    for name, total_flops in top:
        print(f"{name}: {total_flops:.2e} FLOPS")

print_top_flops(flops_cpu, "CPU")
print_top_flops(flops_cuda, "GPU")