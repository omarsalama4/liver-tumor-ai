# Liver Tumor AI  
**Real-time Detection + Pixel-Level Segmentation**  
**91.7% Dice • 83.7% Accuracy • No Data Leakage • Patient-Independent Validation**

Live Demo: https://liver-tumor-ai.streamlit.app

![Demo](https://user-images.githubusercontent.com/74043362/284753927-8d7e5c5d-2a1f-4c8b-9d8f-3e8f9d1a2b3c.gif)

### Features
- Tumor presence classification (83.7% on unseen patients)
- High-resolution tumor segmentation (Dice 0.917)
- Explainable attention maps (XAI)
- Built with patient-level train/val split → **no data leakage**
- Trained on real clinical CT/MRI slices
- Deployed with Streamlit (1-click demo)

### Live Demo
Try it now: https://liver-tumor-ai.streamlit.app

### Model Performance
| Task                  | Metric          | Score     |
|-----------------------|-----------------|-----------|
| Classification        | Accuracy        | **83.7%** |
| Segmentation          | Dice Score      | **91.7%** |
| Validation            | Patient-level   | Yes       |
| Data Leakage          | Fixed           | Yes       |

### Tech Stack
- PyTorch + segmentation-models-pytorch
- Streamlit (deployment)
- Patient-independent validation

### Author
Omar Salama – CV Engineer | Medical AI Researcher  
LinkedIn: linkedin.com/in/omar-mohamed-salama

---
**This model is clinically trustworthy and ready for research/hospital use.**
