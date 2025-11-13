# SARNet
## Author: 
- Junhao Fan (Georgetown University, Department of Data Science & Analytics)
- Wenrui Liang (Tsinghua University, Department of Electrical Engineering)
- Wei-Qiang Zhang (Tsinghua University, Department of Electrical Engineering)

**This research is conducted and leaded by the SATLAB of Tsinghua University**

# Abstract of this paper:
Accurate prediction of remaining useful life (RUL) is essential to enhance system reliability and reduce maintenance risk. Yet many strong contemporary models are fragile around fault onset and opaque to engineers: short, high-energy spikes are smoothed away or misread, fixed thresholds blunt sensitivity, and physics-based explanations are scarce. To remedy this, we introduce SARNet (Spike-Aware Consecutive Validation Framework), which builds on a Modern Temporal Convolutional Network (ModernTCN) and adds spike-aware detection to provide physics-informed interpretability. ModernTCN forecasts degradation-sensitive indicators; an adaptive consecutive threshold validates true spikes while suppressing noise. Failure-prone segments then receive targeted feature engineering (spectral slopes, statistical derivatives, energy ratios), and the final RUL is produced by a stacked RF--LGBM regressor. Across benchmark-ported datasets under an event-triggered protocol, SARNet consistently lowers error compared to recent baselines (RMSE 0.0365, MAE 0.0204) while remaining lightweight, robust, and easy to deploy.

# The reason for this research
In industrial applications, accurately predicting the Remaining Useful Life (RUL) of machinery is crucial for maintenance and reliability. However, many existing models struggle to effectively capture and interpret sudden spikes in degradation data, which are often indicative of impending failures. These spikes can be smoothed out or misinterpreted by traditional models, leading to inaccurate predictions and increased maintenance risks. Additionally, many models lack transparency, making it difficult for engineers to understand the underlying reasons for their predictions.

# Paper link: the preprint of this paper is available at ArXiv
https://arxiv.org/abs/2510.22955
**Code link: the code of this paper is available at this repository.**

# Reference
Please view the reference.bib in this repository for all references used in this paper.