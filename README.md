# Awesome Video Object Segmentation (2022-Present)

![](https://img.shields.io/github/last-commit/gaomingqi/awesome-video-object-segmentation?style=flat-square&colorB=abcdef)

A curated list of awesome video object segmentation (VOS) works since 2022. VOS works before 2022 can be found in our review paper:

>Deep Learning for Video Object Segmentation: A Review [[paper](https://link.springer.com/content/pdf/10.1007/s10462-022-10176-7.pdf)] [[project page](https://github.com/gaomingqi/VOS-Review)]. 

## Contributing

Please feel free to send us pull requests or email (mingqi.gao@outlook.com) to add links :partying_face:.

We employ squares with different colours and abbreviates to highlight different VOS types:

:blue_square: `SVOS`: Semi-Supervised VOS (also termed as One-Shot VOS)

:green_square: `UVOS`: Un-Supervised VOS (also termed as Zero-Shot VOS)

:orange_square: `RVOS`: Referring VOS (also termed as Language-Guided VOS)

## 2023

#### Arxiv 2023
|<img width=87/>||<img width=115/>|
| :-----| :---- | :---- |
|:blue_square: `SVOS`|Co-attention Propagation Network for Zero-Shot Video Object Segmentation|[[paper](https://arxiv.org/pdf/2304.03910.pdf)] [[code](https://github.com/NUST-Machine-Intelligence-Laboratory/HCPN)]|
|:blue_square: `SVOS`|SegGPT: Segmenting Everything In Context (:fire: Generalist model)|[[paper](https://arxiv.org/pdf/2304.03284.pdf)] [[code](https://github.com/baaivision/Painter)]|
||Reliability-Hierarchical Memory Network for Scribble-Supervised Video Object Segmentation|[[paper](https://arxiv.org/pdf/2303.14384.pdf)] [[code](https://github.com/mkg1204/RHMNet-for-SSVOS)]|

#### CVPR 2023

|<img width=87/>||<img width=115/>|
| :-----| :---- | :---- |
|:green_square: `UVOS`|MED-VT: Multiscale Encoder-Decoder Video Transformer with Application to Object Segmentation|[[paper](https://arxiv.org/pdf/2304.05930.pdf)] [[code](https://rkyuca.github.io/medvt/)]|
|:blue_square: `SVOS` |Boosting Video Object Segmentation via Space-time Correspondence Learning| [[paper](https://arxiv.org/pdf/2304.06211.pdf)] [[code](https://github.com/wenguanwang/VOS_Correspondence)]|
|:blue_square: `SVOS` :orange_square: `RVOS`| Universal Instance Perception as Object Discovery and Retrieval (:fire: Generalist model) | [[paper](https://arxiv.org/pdf/2303.06674.pdf)] [[code](https://github.com/MasterBin-IIAU/UNINEXT)]|
|:blue_square: `SVOS` |Two-shot Video Object Segmetnation| [[paper](https://arxiv.org/pdf/2303.12078.pdf)] [[code](https://github.com/yk-pku/Two-shot-Video-Object-Segmentation)]|
|:blue_square: `SVOS`|MobileVOS: Real-Time Video Object Segmentation Contrastive Learning meets Knowledge Distillation |[[paper](https://arxiv.org/pdf/2303.07815.pdf)]|
|:blue_square: `SVOS` |Look Before You Match: Instance Understanding Matters in Video Object Segmentation |[[paper](https://arxiv.org/pdf/2212.06826.pdf)]|

#### AAAI 2023

|<img width=87/>||<img width=115/>|
| :-----| :---- | :---- |
|:blue_square: `SVOS` |Learning to Learn Better for Video Object Segmentation |[[paper](https://arxiv.org/pdf/2212.02112.pdf)]|

#### Journals 2023
|<img width=87/>|||<img width=115/>|
| :-----| :----: | :---- | :---- |
|:orange_square: `RVOS`|TPAMI|VLT: Vision-Language Transformer and Query Generation for Referring Segmentation |[[paper](https://ieeexplore.ieee.org/abstract/document/9932025)]|
|:orange_square: `RVOS` |TPAMI|Local-Global Context Aware Transformer for Language-Guided Video Segmentation |[[paper](https://ieeexplore.ieee.org/abstract/document/10083244)] [[code](https://github.com/leonnnop/Locater)]|

## 2022

#### NeurIPS 2022

|<img width=87/>||<img width=115/>|
| :-----| :---- | :---- |
|:blue_square: `SVOS` |Decoupling Features in Hierarchical Propagation for Video Object Segmentation |[[paper](https://arxiv.org/pdf/2210.09782.pdf)]|
||Self-supervised Amodal Video Object Segmentation |[[paper](https://arxiv.org/pdf/2210.12733.pdf)]|

#### ECCV 2022
|<img width=87/>||<img width=115/>|
| :-----| :---- | :---- |
|:blue_square: `SVOS` |XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model |[[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880633.pdf)] [[code](https://github.com/hkchengrex/XMem)]|
|:blue_square: `SVOS` |BATMAN: Bilateral Attention Transformer in Motion-Appearance Neighboring Space for Video Object Segmentation |[[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890603.pdf)]|
|:blue_square: `SVOS` |Learning Quality-aware Dynamic Memory for Video Object Segmentation |[[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890462.pdf)] [[code](https://github.com/workforai/QDMN)]|
|:blue_square: `SVOS` |Tackling Background Distraction in Video Object Segmentation |[[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820434.pdf)] [[code](https://github.com/suhwan-cho/TBD)]|
|:blue_square: `SVOS` |Global Spectral Filter Memory Network for Video Object Segmentation |[[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890639.pdf)] [[code](https://github.com/workforai/GSFM)]|
|:green_square: `UVOS`  |Hierarchical Feature Alignment Network for Unsupervised Video Object Segmentation |[[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136940584.pdf)]|

#### CVPR 2022
|<img width=87/>||<img width=115/>|
| :-----| :---- | :---- |
|:orange_square: `RVOS` |End-to-End Referring Video Object Segmentation With Multimodal Transformers |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Botach_End-to-End_Referring_Video_Object_Segmentation_With_Multimodal_Transformers_CVPR_2022_paper.pdf)] [[code](https://github.com/mttr2021/MTTR)]|
|:orange_square: `RVOS` |Language As Queries for Referring Video Object Segmentation |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Language_As_Queries_for_Referring_Video_Object_Segmentation_CVPR_2022_paper.pdf)] [[code](https://github.com/wjn922/ReferFormer)]|
|:orange_square: `RVOS` |Language-Bridged Spatial-Temporal Interaction for Referring Video Object Segmentation |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Language-Bridged_Spatial-Temporal_Interaction_for_Referring_Video_Object_Segmentation_CVPR_2022_paper.pdf)] [[code](https://github.com/dzh19990407/LBDT)]|
|:orange_square: `RVOS` |Multi-Level Representation Learning With Semantic Alignment for Referring Video Object Segmentation |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Multi-Level_Representation_Learning_With_Semantic_Alignment_for_Referring_Video_Object_CVPR_2022_paper.pdf)]|
|:blue_square: `SVOS` |Recurrent Dynamic Embedding for Video Object Segmentation |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Recurrent_Dynamic_Embedding_for_Video_Object_Segmentation_CVPR_2022_paper.pdf)] [[code](https://github.com/Limingxing00/RDE-VOS-CVPR2022)]|
|:blue_square: `SVOS` |Accelerating Video Object Segmentation With Compressed Video |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Accelerating_Video_Object_Segmentation_With_Compressed_Video_CVPR_2022_paper.pdf)] [[code](https://github.com/kai422/CoVOS)]|
|:blue_square: `SVOS` |SWEM: Towards Real-Time Video Object Segmentation With Sequential Weighted Expectation-Maximization |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_SWEM_Towards_Real-Time_Video_Object_Segmentation_With_Sequential_Weighted_Expectation-Maximization_CVPR_2022_paper.pdf)]|
|:blue_square: `SVOS` |Per-Clip Video Object Segmentation |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Per-Clip_Video_Object_Segmentation_CVPR_2022_paper.pdf)] [[code](https://github.com/pkyong95/PCVOS)]|
||Wnet: Audio-Guided Video Object Segmentation via Wavelet-Based Cross-Modal Denoising Networks |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Pan_Wnet_Audio-Guided_Video_Object_Segmentation_via_Wavelet-Based_Cross-Modal_Denoising_Networks_CVPR_2022_paper.pdf)] [[code](https://github.com/asudahkzj/Wnet)]|
||YouMVOS: An Actor-Centric Multi-Shot Video Object Segmentation Dataset |[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_YouMVOS_An_Actor-Centric_Multi-Shot_Video_Object_Segmentation_Dataset_CVPR_2022_paper.pdf)] [[DATASET](https://donglaiw.github.io/proj/youMVOS/)]|

#### AAAI 2022
|<img width=87/>||<img width=115/>|
| :-----| :---- | :---- |
|:blue_square: `SVOS` |Siamese Network with Interactive Transformer for Video Object Segmentation |[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20009)] [[code](https://github.com/LANMNG/SITVOS)]|
|:blue_square: `SVOS` |Reliable Propagation-Correction Modulation for Video Object Segmentation |[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20200)]|
|:orange_square: `RVOS` |You Only Infer Once: Cross-Modal Meta-Transfer for Referring Video Object Segmentation |[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20017)]|
|:green_square: `UVOS` |Iteratively Selecting an Easy Reference Frame Makes Unsupervised Video Object Segmentation Easier |[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20011)]|

#### Journals 2023

|<img width=87/>|||<img width=115/>|
| :-----| :----: | :---- | :---- |
|:blue_square: `SVOS` |TPAMI|Video Object Segmentation Using Kernelized Memory Network With Multiple Kernels |[[paper](https://ieeexplore.ieee.org/document/9745367)]|
|:blue_square: `SVOS` |TIP|From Pixels to Semantics: Self-Supervised Video Object Segmentation With Multiperspective Feature Mining |[[paper](https://ieeexplore.ieee.org/document/9875116)]|
|:blue_square: `SVOS` |TIP|Delving Deeper Into Mask Utilization in Video Object Segmentation |[[paper](https://ieeexplore.ieee.org/document/9904497)]|
|:blue_square: `SVOS` |TIP|Adaptive Online Mutual Learning Bi-Decoders for Video Object Segmentation |[[paper](https://ieeexplore.ieee.org/document/9942927)]|

