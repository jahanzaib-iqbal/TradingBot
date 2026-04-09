import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from strategy.smart_money_strategy import SmartMoneyStrategy

def make_df(closes, noise=0.1, seed=21):
    rng = np.random.default_rng(seed)
    n   = len(closes)
    hi  = closes + rng.uniform(0.5 * noise, 1.5 * noise, n)
    lo  = closes - rng.uniform(0.5 * noise, 1.5 * noise, n)
    op  = closes + rng.uniform(-0.3 * noise, 0.3 * noise, n)
    hi  = np.maximum(hi, np.maximum(op, closes))
    lo  = np.minimum(lo, np.minimum(op, closes))
    vol = rng.integers(500, 1500, n).astype(float)
    start = datetime(2024, 3, 1, tzinfo=timezone.utc)
    return pd.DataFrame({
        "time":        [start + timedelta(minutes=15 * i) for i in range(n)],
        "open":        op.round(2),
        "high":        hi.round(2),
        "low":         lo.round(2),
        "close":       closes.round(2),
        "tick_volume": vol,
    })

n = 200
closes = np.full(n, 2100.0, dtype=float)
closes[:150] = np.linspace(2100, 2180, 150)
closes[58] = 2140; closes[59] = 2135; closes[60] = 2130.0
closes[61] = 2135; closes[62] = 2140
closes[120] = 2165; closes[121] = 2169; closes[122] = 2175; closes[123] = 2181
closes[150] = 2124; closes[151] = 2132; closes[152] = 2135
closes[153] = 2140; closes[154] = 2148; closes[155] = 2155
closes[156:] = np.linspace(2155, 2167, n - 156)
df = make_df(closes, noise=0.1, seed=21)
df.at[120, "open"]  = 2168; df.at[120, "close"] = 2160
df.at[120, "high"]  = 2169.5; df.at[120, "low"] = 2159
df.at[150, "open"]  = 2175; df.at[150, "low"]   = 2124
df.at[150, "high"]  = 2177; df.at[150, "close"] = 2132

smc = SmartMoneyStrategy()
smc.MIN_CONFIDENCE = 0.10
smc.OB_MIN_BODY_RATIO = 0.35
smc.OB_MIN_IMPULSE_ATR = 0.1

sw  = smc._find_swings(df)
ms  = smc.detect_market_structure(df, sw)
print("Trend bias:", smc.get_trend_bias(ms))
pools = smc.detect_liquidity_pools(df, sw)

min_low    = float(df["low"].min())
max_high   = float(df["high"].max())
final_close = float(df["close"].iloc[-1])
print(f"min_low={min_low:.2f}  max_high={max_high:.2f}  final_close={final_close:.2f}")

low_pools = [p for p in pools if p.pool_type in ("LOW", "EQUAL_LOW")]
print(f"LOW pools: {len(low_pools)}")
for p in low_pools:
    print(f"  {p.pool_type:<12} price={p.price:.2f}  bar={p.bar_index}  swept={p.swept}")

swept = [p for p in low_pools if p.swept]
print(f"Swept LOW pools: {len(swept)}")

atr = smc._atr(df)
obs = smc.detect_order_blocks(df, atr)
print(f"OBs: {len(obs)}")
for ob in obs[:3]:
    print(f"  {ob.ob_type}  bar={ob.bar_index}  zone=[{ob.ob_low:.2f}, {ob.ob_high:.2f}]  valid={ob.valid}")

setups = smc.scan(df)
print(f"\nSetups (min_conf=0.10): {len(setups)}")
for s in setups:
    print(f"  {s}")
