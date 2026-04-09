# Strategy Documentation — Smart Money Concepts (SMC)

A technical deep-dive into the trading methodology implemented in the AntiGravity Gold Bot.

---

## Overview

The bot uses a **Smart Money Concepts (SMC)** framework to identify institutional trading activity in XAUUSD. SMC focuses on how large market participants (banks, hedge funds) accumulate and distribute positions, creating predictable price patterns that can be exploited.

The pipeline is multi-timeframe:

| Timeframe | Role |
|-----------|------|
| **H4** | Macro bias — overall market structure and regime |
| **H1** | Setup identification — where is the SMC setup? |
| **M15** | Entry refinement — precise entry within the setup zone |

---

## 1. Market Regime Classification

Before any setup is evaluated, the overall market regime is determined on the **H4** chart.

### RegimeLabel Enum

```
TRENDING_UP    — Higher Highs (HH) + Higher Lows (HL) + ADX ≥ threshold
TRENDING_DOWN  — Lower Lows (LL) + Lower Highs (LH) + ADX ≥ threshold
RANGING        — No clear structure + ADX < threshold
UNKNOWN        — Insufficient data
```

### Classification Logic

1. **ADX check**: If `ADX < ADX_TREND_THRESHOLD (25)` → `RANGING`
2. **Swing structure**: Detect last 3 significant swing points
   - HH + HL pattern → `TRENDING_UP`
   - LL + LH pattern → `TRENDING_DOWN`
   - Mixed / overlapping → `RANGING`

### Signal gates by regime

| Regime | Allowed signal directions |
|--------|--------------------------|
| `TRENDING_UP` | BUY only |
| `TRENDING_DOWN` | SELL only |
| `RANGING` | Both (with reduced confidence score) |

---

## 2. EMA Trend Filter

Applied on both H4 and H1. Three EMAs are computed: **EMA 20** (fast), **EMA 50** (slow), **EMA 200** (macro).

### Bullish alignment (BUY bias)
```
Price > EMA_20 > EMA_50 > EMA_200
EMA_20 slope angle > EMA_MIN_SLOPE_ANGLE
```

### Bearish alignment (SELL bias)
```
Price < EMA_20 < EMA_50 < EMA_200
EMA_20 slope angle < -EMA_MIN_SLOPE_ANGLE
```

**Partial alignment** (e.g., price above EMA 20 but below EMA 200) reduces the confidence score rather than outright blocking the signal.

---

## 3. Break of Structure (BOS) & Change of Character (ChoCH)

### Break of Structure (BOS)
A **continuation** signal — the market breaks a previous structural high (bullish BOS) or low (bearish BOS) in the direction of the existing trend.

```
Bullish BOS: price closes above the most recent swing high
Bearish BOS: price closes below the most recent swing low
```

### Change of Character (ChoCH)
A **reversal** signal — the market breaks structure against the prevailing trend, indicating a potential regime shift.

```
Bullish ChoCH: bearish market breaks above a recent swing high
Bearish ChoCH: bullish market breaks below a recent swing low
```

The bot primarily trades **BOS continuations** with the trend. ChoCH setups are permitted only when the confluence score is very high (≥ 0.75).

---

## 4. Order Blocks (OB)

An Order Block is the **last opposing candle before a strong impulsive move**. It represents an area where institutional orders were placed and price is likely to return to for re-testing.

### Bullish Order Block
- The last **bearish** (red) candle immediately before a bullish impulse (3+ consecutive up candles)
- Entry zone: the full range of that candle (`low` to `high`)
- Invalidated if price closes below the OB low

### Bearish Order Block
- The last **bullish** (green) candle immediately before a bearish impulse (3+ consecutive down candles)
- Entry zone: the full range of that candle (`low` to `high`)
- Invalidated if price closes above the OB high

### Quality filters
- Candle body / range ≥ `OB_MIN_BODY_RATIO (0.5)` — avoids doji-like weak blocks
- Impulse must be at least `OB_MIN_IMPULSE_CANDLES (3)` bars

---

## 5. Fair Value Gaps (FVG)

A Fair Value Gap (also called an **imbalance** or **inefficiency**) is a three-candle pattern where price moved so fast that there is a gap between candle[n-1] and candle[n+1].

### Bullish FVG
```
candle[n-1].high  <  candle[n+1].low
```
The gap between these two prices is the FVG zone.

### Bearish FVG
```
candle[n-1].low  >  candle[n+1].high
```

### Quality filter
- FVG size (in price points) ≥ `FVG_MIN_ATR_FRACTION × ATR` — avoids tiny insignificant gaps

FVGs overlapping with an OB create the highest-confluence setups.

---

## 6. Liquidity Pools

Institutional players target areas where **retail stop-losses cluster**. These are the liquidity pools the bot maps:

| Pool Type | Definition |
|-----------|-----------|
| **Equal Highs** | Two or more swing highs at approximately the same price |
| **Equal Lows** | Two or more swing lows at approximately the same price |
| **Previous Session High** | High of the London or previous NY session |
| **Previous Session Low** | Low of the London or previous NY session |
| **Round Numbers** | $2300, $2350, $2400, etc. (psychological levels) |

Liquidity pools serve as **Take Profit targets** (the bot expects price to sweep through them) and as confluence factors when price is approaching a valid OB.

---

## 7. Confluence Scoring

Each setup receives a composite score from **0.0 to 1.0**. Only setups above `CONFIDENCE_THRESHOLD (0.68)` generate a signal.

| Factor | Max points | Conditions |
|--------|-----------|------------|
| Regime alignment | 0.25 | Direction matches `RegimeLabel` |
| Order Block quality | 0.25 | Valid OB within price zone + body ratio ≥ 0.5 |
| FVG overlap | 0.20 | FVG overlaps with OB zone |
| BOS / ChoCH confirmation | 0.15 | Recent structural break in signal direction |
| Liquidity target | 0.15 | Clear liquidity pool as TP target |

**Examples**:
- Regime ✓ + Strong OB ✓ + FVG ✓ + BOS ✓ + Liquidity ✓ → 1.0 (perfect setup)
- Regime ✓ + Weak OB ✓ + No FVG + BOS ✓ + No liquidity → ~0.55 (below threshold, rejected)

---

## 8. Entry, Stop Loss & Take Profit Calculation

### Entry
- Mid-point of the Order Block zone (or FVG zone if OB not present)
- On M15 for precision: wait for a confirmation candle (e.g., pin bar, engulfing) within the zone

### Stop Loss
```
BUY  SL = OB_low  -  (ATR × ATR_SL_MULTIPLIER)
SELL SL = OB_high +  (ATR × ATR_SL_MULTIPLIER)
```

### Take Profit 1 (partial)
```
BUY  TP1 = entry + (ATR × ATR_TP1_MULTIPLIER)
SELL TP1 = entry - (ATR × ATR_TP1_MULTIPLIER)
```

### Take Profit 2 (final)
```
BUY  TP2 = nearest liquidity pool above entry
SELL TP2 = nearest liquidity pool below entry
```

If TP2 does not yield at least `MIN_RR_RATIO (2.0)`, the signal is discarded.

---

## 9. Signal Quality Score

After all factors are calculated, the final `TradingSignal` carries:

| Field | Description |
|-------|-------------|
| `confluence_score` | 0.0 – 1.0 weighted SMC score |
| `risk_reward_ratio` | Actual R:R of the setup |
| `regime` | Market regime label |
| `session` | Session during which signal was generated |

---

## Future Enhancements

- **ML signal scorer**: Train a scikit-learn classifier on backtest trade logs to predict outcome probability
- **Wyckoff integration**: Add accumulation/distribution phase detection
- **Volume profile**: Incorporate volume-at-price analysis (requires tick data or VSA libraries)
- **Multi-symbol expansion**: Extend pipeline to EURUSD, US30 (correlated assets for Gold bias)

---

*Last updated: April 2026 | AntiGravity Trading Bot v1.0*
