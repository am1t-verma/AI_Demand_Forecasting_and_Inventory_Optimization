# Food & Restaurant Services
## AI Demand Forecasting and Inventory Optimization Roadmap

### 1. Project Objective
Predict weekly `num_orders` for each `(center_id, meal_id)` combination so operations can:
- reduce stockouts (lost sales),
- reduce overproduction (waste),
- improve procurement and kitchen planning.

### 2. Data Assets and Roles
- `train.csv`: historical weekly demand with pricing and promotion signals.
- `test.csv`: future rows where `num_orders` must be predicted.
- `fulfilment_center_info.csv`: center-level business context.
- `meal_info.csv`: meal-level product context.
- `sample_submission.csv`: required output format (`id`, predicted `num_orders`).

### 3. Business-to-ML Mapping
- `week`: when demand happens (trend + seasonality signal).
- `center_id`: where demand happens (local behavior, center capacity).
- `meal_id`: what is demanded (item preference pattern).
- `checkout_price`, `base_price`: price behavior and discount effect.
- `emailer_for_promotion`, `homepage_featured`: demand uplift drivers.
- `num_orders`: target to forecast.

### 4. Validation Strategy (Critical)
Use a time-based split only:
- Train on earlier weeks.
- Validate on later weeks.
- Never random split (causes leakage in forecasting tasks).

Recommended approach:
- Start with one holdout window (example: last 10-20 weeks).
- Upgrade to rolling/expanding time validation once baseline is stable.

### 5. Data Quality and Sanity Checks
Before modeling, verify:
- unique key behavior for `(week, center_id, meal_id)`,
- missing values by column,
- unexpected negative or zero pricing patterns,
- value ranges for promotions (`0/1` consistency),
- overlap of all IDs between train/test and metadata tables.

### 6. Feature Engineering Roadmap
Build features in layers:

Layer A: Core structural features
- center metadata: `city_code`, `region_code`, `center_type`, `op_area`
- meal metadata: `category`, `cuisine`

Layer B: Price and discount features
- discount amount (`base_price - checkout_price`)
- discount ratio
- price spread/volatility over recent weeks

Layer C: Temporal features
- week index transforms (cyclic or segmented)
- recency windows
- seasonal pattern proxies

Layer D: Historical demand signals (leakage-safe)
- lag demand features per `(center_id, meal_id)`
- rolling means/medians/std over past windows
- recent trend direction

Layer E: Interaction features
- promo x discount
- center type x category
- cuisine x region

### 7. Modeling Ladder
Progress in this order:
1. Naive baseline (simple historical rule).
2. Regularized linear/tree baseline.
3. Gradient boosting model (strong tabular performer).
4. Ensemble of best complementary models (optional).

Rule: only move to next model after error analysis justifies it.

### 8. Error Analysis Framework
Slice validation error by:
- center,
- meal category/cuisine,
- promotion flag,
- discount intensity,
- high-demand vs low-demand weeks.

Goal: identify where model fails and design targeted features.

### 9. Inventory Optimization Translation
After obtaining forecasts:
- convert predicted demand to procurement quantity using safety buffer,
- define service-level target (stockout tolerance),
- calibrate buffer per meal volatility and perishability.

Outputs to prepare:
- forecast table (`id`, predicted `num_orders`)
- center-meal weekly procurement recommendation
- uncertainty/risk band for operational planning

