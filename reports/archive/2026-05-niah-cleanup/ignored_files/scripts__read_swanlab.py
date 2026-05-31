"""通过 SwanLab Api 读取训练指标数据并分析趋势"""
import swanlab
import time

api = swanlab.Api()
run = api.run("Saika/DSRA/h55cpbbpnevasgejb539b")

print(f"Run: {run.name}")
print(f"State: {run.state}")
print()

# 重试获取指标
for attempt in range(3):
    try:
        metrics = run.metrics(keys=["train/loss", "train/ppl"])
        print(f"Metrics shape: {metrics.shape}")
        print(f"Metrics columns: {list(metrics.columns)}")
        if not metrics.empty:
            print()
            print("Loss 趋势:")
            print(metrics.to_string())
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(2)

print()

# 尝试获取门控指标
for attempt in range(3):
    try:
        gate_metrics = run.metrics(keys=["gate/avg_st_ratio", "gate/avg_st_weight", "gate/avg_mh_weight"])
        print(f"Gate metrics shape: {gate_metrics.shape}")
        if not gate_metrics.empty:
            print()
            print("门控权重趋势:")
            print(gate_metrics.to_string())
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(2)

print()

# 尝试获取 slot 指标
for attempt in range(3):
    try:
        slot_metrics = run.metrics(keys=["slot/avg_token_gate", "slot/avg_write_mass", "slot/avg_usage"])
        print(f"Slot metrics shape: {slot_metrics.shape}")
        if not slot_metrics.empty:
            print()
            print("Slot 写入趋势:")
            print(slot_metrics.to_string())
        else:
            print("Slot 指标为空（可能是新训练还没有记录到 slot 数据）")
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(2)
