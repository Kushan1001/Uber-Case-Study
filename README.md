# Uber Ride Cancellation Prediction

A case study using Uber's NCR ride bookings dataset (150,000 rides) to predict whether a booking will complete or get cancelled, then using that model to figure out where the business should actually spend money to reduce cancellations.

## The data

150,000 bookings, 21 columns — date/time, customer and booking IDs, vehicle type, pickup/drop locations, VTAT/CTAT, ratings, cancellation reasons, payment method, etc. Sorted by date and split 80/20 into train (120,000) and test (30,000) so the split is chronological, not random — the model is tested on rides that happened after the ones it trained on.

A few columns had heavy missingness (things like cancellation reason or incomplete ride reason are only populated when that thing actually happened, so 120k+ nulls there is expected, not a data quality problem).

## Framing the problem

Target: booking completed (1) vs not completed (0).

Before modeling, I dropped columns that would leak the outcome — ride distance, booking value, driver/customer ratings, payment method, and the cancellation/incomplete reason fields all only exist *after* a ride resolves, so training on them would let the model "cheat."

Feature engineering added: hour/day/weekend flags, a time-of-day bucket (morning peak, midday, evening rush, night), pickup-location booking density, per-customer historical cancellation rate, and target-encoded pickup/drop locations.

## Models tried

Three models, each wrapped in a pipeline with SMOTE (to handle the class imbalance — roughly 38% of test-set bookings are cancellations) and tuned with RandomizedSearchCV on F1:

| Model | Accuracy | Precision | Recall | F1 | False Positives | False Negatives | Business Cost (₹) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.925 | 0.896 | 0.995 | 0.943 | 2,144 | 93 | 10,813 |
| Random Forest | 0.929 | 0.902 | 0.995 | 0.946 | 2,025 | 92 | 998 |
| LightGBM | 0.994 | 0.995 | 0.995 | 0.995 | 163 | 93 | 578 |

Business cost here is a made-up but reasonable weighting: a false positive (predicting a ride completes when it actually cancels) costs 5x more than a false negative, since missing a real cancellation risk is more expensive to the business than being overly cautious. LightGBM wins by a wide margin on every metric, not just the composite cost, so it's the model I'd actually ship.

I also tried optimizing the decision threshold to minimize that business cost, but for LightGBM it made no difference — the model's predicted probabilities are so polarized (mostly near 0 or near 1) that moving the threshold around doesn't change any predictions. That's actually a good sign for model confidence, not a bug.

## What the model gets wrong

163 false positives and 93 false negatives out of 30,000 test predictions (an 0.85% error rate). Looking at where the false positives cluster: it's not random noise, it correlates with predictable things like specific pickup zones and time slots rather than model instability.

## Turning the model into two business decisions

Rather than stop at "here's a model," I ran two rough what-if simulations using the model's output:

**1. Driver incentives in low-supply zones** — zones with a completion rate under 60% (48 zones, 8,145 bookings in the test set). If a ₹50/ride incentive lifts completion by 10 points there, estimated net benefit is **₹81,450** (₹203,625 revenue gain against ₹122,175 in incentive cost).

**2. Cancellation fee for high-risk bookings** — bookings the model flags with >80% cancellation probability (11,312 in the test set, which actually cancel 99.2% of the time). A ₹20 fee with an assumed 25% behavior change, netted against a 5% customer churn risk from the fee, comes out to an estimated **₹784,632/month** net benefit (ROI ~925%).

Both of these depend on assumptions I made up (behavior change rate, churn rate, average ride value) rather than measured effects, so they're illustrative rather than a guarantee — the value is in showing how the classifier's output plugs into an actual revenue decision, not the exact rupee figures.

## High-risk segments

Three segments account for about 60% of all cancellations in the test set:
- Repeat cancellers (customer cancellation rate >40%): small group (145 bookings) but concentrated
- Peak hours (morning/evening rush): 13,709 bookings, 46% of all cancellations
- Bottom-quartile pickup zones by completion rate: 7,482 bookings, ~25% of cancellations

## Operational / ethical checks

- **Latency**: single-prediction inference with the tuned LightGBM model came in at 6.48ms, well within range for a live API call.
- **Geographic fairness**: false positive rate was checked across the best- and worst-performing zones (1.21% vs 1.64%) — a small enough gap that the model doesn't look like it's systematically penalizing any particular area, but this is worth monitoring in production rather than treating as settled.

## Stack

pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), LightGBM, matplotlib, seaborn

## Files

- `Uber_Case_Study.ipynb` — full notebook: EDA, cleaning, feature engineering, three models, threshold optimization, error analysis, business simulations, fairness checks
