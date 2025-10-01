## RL Dopamine Demo — Reward Prediction Errors (RPE)

This small project demonstrates how temporal-difference (TD) error can act like a dopamine reward prediction error (RPE).

Interpretation: unexpected reward produces positive RPE peaks; omission produces negative dips.


## Results

### Returns
Learning performance across conditions (baseline, unexpected reward, omission):

![Returns](graphs/all_returns_comparison.png)

### TD Errors (dopamine-like signals)
Average TD error aligned to reward events:

![TD Errors](graphs/all_td_errors_comparison.png)

- **Positive burst** when unexpected reward appears  
- **Negative dip** when expected reward is omitted  
- **Baseline** resembles stabilized dopamine response  

---

### Individual Experiment Traces

#### Baseline
![Baseline](graphs/aligned_tds_baseline.png)

#### Unexpected Reward
![Unexpected Reward](graphs/aligned_tds_unexpected_reward.png)

#### Reward Omission
![Reward Omission](graphs/aligned_tds_reward_omission.png)

---
