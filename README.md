# Awesome Dynamic Networks and Conditional Computation


<br><br>

> **Upcoming ICML 2022 on Dynamic Neural Networks! <https://dynn-icml2022.github.io/> on Friday, July 22.**

<br><br>


Overview of conditional computation and dynamic CNNs for computer vision, focusing on reducing computational cost of existing network architectures. In contrast to static networks, dynamic networks disable parts of the network based on the input image, at inference time. This can save computations and speed up inference, for example by processing easy images with fewer operations. Note that this list mainly focuses on methods reducing the computational cost of existing models (e.g. ResNet models), and does not list all methods that use dynamic computation for custom architectures.



**This list is growing every day. If a method is missing or listed incorrectly, let me know by making a GitHub issue or pull request!**

[Here](https://github.com/MingSun-Tse/EfficientDNNs) is a list with more static and dynamic methods for efficient CNNs.

## Background 
Methods have three important distinguishing factors:
* The method's architecture, e.g. skipping layers or pixels, and whether these run-or-skip decisions are the result of a separate policy network, a submodule in the network or another mechanism.
* The way of training the policy, e.g. using reinforcement learning, the gradient estimator such as Gumbel-Softmax or a custom approach.
* The implementation of the method, and whether the method can be executed efficiently on existing platforms (i.e. whether the method speeds up inference, or only reduces the theoretical amount of computations)

**Metrics**:
Most methods demonstrate performance with the reduction in computations (i.e. measured in floating point operations, FLOPS) compared to the loss in accuracy. 
Methods typically show figures where  baseline models of different complexities (e.g. by reducing the number of channels) are compared to the method applied to the largest model with different cost savings.

Note that many works express computational complexity in FLOPS, even though the given numbers are actually multiply-accumulate operations (MACs), and GMACs = 0.5 * GFLOPs (see https://github.com/sovrasov/flops-counter.pytorch/issues/16 ). Some recent works therefore use GMAC instead of GFLOP to avoid ambiguity.


**Tags used below**:
Note: tags are incomplete
<!-- * RL: reinforcement learning
* GS: Gumbel-SoftMax
* IN1K: ImageNet -->
* VID: Video processing

## Surveys / overviews
* **Dynamic Neural Networks: A Survey** (Arxiv 2021) [[pdf]](https://arxiv.org/pdf/2102.04906)
Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, Yulin Wang


## Methods 
### Depth-based methods

Early-exit methods have separate output branches to apply more or fewer layers.

* **BranchyNet: Fast inference via early exiting from deep neural networks** (ICPR2016) [[pdf]](https://ieeexplore.ieee.org/iel7/7893644/7899596/07900006.pdf?casa_token=6Rw2FfYo7GkAAAAA:oTMVtfqbSCwdNlZp9uZBldholjMd52rMGma1WNiASyWhIrShrYifvgactwhUgioAt3Lu2Un7t6mt) [[chainer]](https://github.com/kunglab/branchynet)  
Teerapittayanon S, McDanel B, Kung HT 
* **Conditional Deep Learning for Energy-Efficient and Enhanced Pattern Recognition** (DATE2016) [[pdf]](https://ieeexplore.ieee.org/iel7/7454909/7459269/07459357.pdf?casa_token=RRkd-SqJBp0AAAAA:SpMrgFxhFU3NWFVFdQoDFrVk0OP1LYGf_lj0xcyAe1GeqBxQCcafwDV0G2F-SXWBy_zuQc2GS5B9)  
P. Panda, A. Sengupta, and K. Roy
* **Adaptive Neural Networks for Efficient Inference**  (ICML2017) [[pdf]](https://arxiv.org/pdf/1702.07811) [[GitHub no code]](https://github.com/tolga-b/ann)  
T. Bolukbasi, J. Wang, O. Dekel, and V. Saligrama
* **Dynamic computational time for visual attention**  (ICCV2017 workshop) [[pdf]](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w18/Li_Dynamic_Computational_Time_ICCV_2017_paper.pdf)  [[torch lua]](https://github.com/baidu-research/DT-RAM)  
Li, Z., Yang, Y., Liu, X., Zhou, F., Wen, S. and Xu, W.
* **DynExit: A Dynamic Early-Exit Strategy for Deep Residual Networks** (SiPS2019) [[pdf]](https://ieeexplore.ieee.org/iel7/9006748/9020312/09020551.pdf?casa_token=idnY9XZfHY8AAAAA:IqKAN95mZo4NxRiua7E4bfLUClg2pkb4ZRTTM4x17zbB6OI08kmSfn-CGCmpKrz4rePAIvXFby3m)  
M. Wang, J. Mo, J. Lin, Z. Wang, and L. Du
* **Improved Techniques for Training Adaptive Deep Networks** (ICCV2019) [[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Improved_Techniques_for_Training_Adaptive_Deep_Networks_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/pawopawo/IMTA)  
H. Li, H. Zhang, X. Qi, Y. Ruigang, and G. Huang
* **Early-exit convolutional neural networks** (thesis 2019) [[pdf]](http://etd.lib.metu.edu.tr/upload/12622986/index.pdf)  
E. Demir
* **Efficient adaptive inference for deep convolutional neural networks using hierarchical early exits** (Pattern Recognition 2020) [[pdf]](https://www.sciencedirect.com/science/article/pii/S0031320320301497?casa_token=EJ-ZeTUJwP8AAAAA:bcZVeD_hPG59_v0DidF1CbSvXuhajEBmBH5OWACtaCCJiQvIcPBgi7z9TPrfSURyZ-OUJYDE3l4I)  
N. Passalis, J. Raitoharju, A. Tefas, and M. Gabbouj
* **Triple wins: Boosting accuracy, robustness and efficiency together by enabling input-adaptive inference** (ICLR2020) [[pdf]](https://arxiv.org/pdf/2002.10025.pdf) [[pytorch]](https://github.com/VITA-Group/triple-wins)  
Hu TK, Chen T, Wang H, Wang Z. 
* **FrameExit: Conditional Early Exiting for Efficient Video Recognition** [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghodrati_FrameExit_Conditional_Early_Exiting_for_Efficient_Video_Recognition_CVPR_2021_paper.pdf)
Ghodrati, A., Bejnordi, B. E., & Habibian, A.  
[VID]

Skipping layers conditioned on the input image. For instance, easy images require fewer layers than complex ones:

*  **Adaptive Computation Time for Recurrent Neural Networks** (NIPS 2016 Deep Learning Symposium) [[pdf]](https://arxiv.org/pdf/1603.08983) [[unofficial pytorch]](https://github.com/zphang/adaptive-computation-time-pytorch)  
A. Graves
* **Convolutional Networks with Adaptive Inference Graphs**  (ECCV2018) [[pdf]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Andreas_Veit_Convolutional_Networks_with_ECCV_2018_paper.pdf) [[Pytorch]](https://github.com/andreasveit/convnet-aig)  
A. Veit and S. Belongie
* **SkipNet: Learning Dynamic Routing in Convolutional Networks** (ECCV2018) [[pdf]](https://arxiv.org/pdf/1711.09485.pdf) [[Pytorch]](https://github.com/ucbdrive/skipnet)    
X. Wang, F. Yu, Z.-Y. Dou, T. Darrell, and J. E. Gonzalez
* **BlockDrop: Dynamic Inference Paths in Residual Networks** (CVPR2018) [[pdf]](https://arxiv.org/pdf/1711.08393.pdf) [[Pytorch]](https://github.com/Tushar-N/blockdrop>)  
Zuxuan Wu*, Tushar Nagarajan*, Abhishek Kumar, Steven Rennie, Larry S. Davis, Kristen Grauman, and Rogerio Feris
* **Dynamic Multi-path Neural Network** (Arxiv2019)  [[pdf]](https://arxiv.org/pdf/1902.10949)  
Su, Y., Zhou, S., Wu, Y., Su, T., Liang, D., Liu, J., Zheng, D., Wang, Y., Yan, J. and Hu, X.
* **Energynet: Energy-efficient dynamic inference** (2018) [[pdf]](https://openreview.net/pdf?id=Syxp2bgKoX)  
Wang, Yue, et al.
* **Dual dynamic inference: Enabling more efficient, adaptive and controllable deep inference** (IEEE Journal of Selected Topics in Signal Processing 2020) [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9028245&casa_token=IeCWKAjjijsAAAAA:rMsL1jES4Z6CeoQ-zF1IUsr8EZolkPZ_xhcEDd8sb8IBLo_nMt3j7U9wtJQNo8LAzYl6TuKb71zp&tag=1)  
Wang Y, Shen J, Hu TK, Xu P, Nguyen T, Baraniuk RG, Wang Z, Lin Y. 
* **CoDiNet: Path Distribution Modeling with Consistency and Diversity for Dynamic Routing** (TPAMI 2021) [[pdf]](https://ieeexplore.ieee.org/document/9444192)  



Executes some layers multiple times ('recursively') based on complexity:

* **IamNN: Iterative and Adaptive Mobile Neural Network for Efficient Image Classification** (ICLR2018 Workshop) [[pdf]](https://arxiv.org/pdf/1804.10123)  
S. Leroux, P. Molchanov, P. Simoens, B. Dhoedt, T. Breuel, and J. Kautz
* **Dynamic recursive neural network** (CPVR2019) [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Dynamic_Recursive_Neural_Network_CVPR_2019_paper.pdf)  
Guo, Q., Yu, Z., Wu, Y., Liang, D., Qin, H., and Yan, J.  

### Channel-based methods
Channel-based methods execute specific channels to reduce computational complexity.
* **Estimating or propagating gradients through stochastic neurons for conditional computation** [[pdf]](https://arxiv.org/pdf/1308.3432.pdf)  
Bengio Y, Léonard N, Courville A.
* **Runtime Neural Pruning** (NIPS2017) [[pdf]](http://papers.nips.cc/paper/6813-runtime-neural-pruning.pdf)  
J. Lin, Y. Rao, J. Lu, and J. Zhou
* **Dynamic Channel Pruning: Feature Boosting and Suppression** (Arxiv2018) [[pdf]](https://arxiv.org/pdf/1810.05331) [[tensorflow]](https://github.com/deep-fry/mayo) [[unoffical pytorch]](https://github.com/yulongwang12/pytorch-fbs)  
 X. Gao, Y. Zhao, Ł. Dudziak, R. Mullins, and C. Xu.
* **Channel Gating Neural Networks** (NIPS2019) [[pdf]](https://papers.nips.cc/paper/8464-channel-gating-neural-networks.pdf) [[pytorch]](https://github.com/cornell-zhang/dnn-gating)  
W. Hua, Y. Zhou, C. M. De Sa, Z. Zhang, and G. E. Suh
* **You Look Twice: GaterNet for Dynamic Filter Selection in CNNs**  (CVPR2019) [[pdf]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_You_Look_Twice_GaterNet_for_Dynamic_Filter_Selection_in_CNNs_CVPR_2019_paper.pdf)  
Z. Chen, Y. Li, S. Bengio, and S. Si
* **Runtime Network Routing for Efficient Image Classification** (TPAMI2019) [[pdf]](https://ieeexplore.ieee.org/iel7/34/8824149/08510920.pdf?casa_token=hMXAytJvXQ4AAAAA:4e4H69Y-K9XlzXxIR-tr0e76MQomeRPAnOiuKXLY2jIdR4_Lg3ZkyF5WRt14942gFd5MSoafy16j)  
Y. Rao, J. Lu, J. Lin, and J. Zhou
* **Dynamic Neural Network Channel Execution for Efficient Training** (BMVC2019) [[pdf]](https://arxiv.org/pdf/1905.06435)  
S. E. Spasov and P. Lio
* **Learning Instance-wise Sparsity for Accelerating Deep Models** (IJCAI2019) [[pdf]](https://arxiv.org/pdf/1907.11840.pdf)  
Liu C, Wang Y, Han K, Xu C, Xu C.
* **Batch-Shaping for Learning Conditional Channel Gated Networks** (ICLR2020) [[pdf]](https://arxiv.org/pdf/1907.06627.pdf)  
BE Bejnordi, T Blankevoort, M Welling
* **Dynamic slimmable network** (CVPR2021) [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dynamic_Slimmable_Network_CVPR_2021_paper.pdf) [[pytorch]](https://github.com/changlin31/DS-Net)  
Li, Changlin, et al.
* **Dynamic Slimmable Denoising Network.** (2021) [[pdf]](https://arxiv.org/pdf/2110.08940)
Jiang, Zutao, Changlin Li, Xiaojun Chang, Jihua Zhu, and Yi Yang
* **DS-Net++: Dynamic Weight Slicing for Efficient Inference in CNNs and Transformers** (2021) [[pdf]](https://arxiv.org/pdf/2109.10060)
Li, C., Wang, G., Wang, B., Liang, X., Li, Z., & Chang, X.

* **Borrowing from yourself: Faster future video segmentation with partial channel update** (2022) [[pdf]](https://arxiv.org/abs/2202.05748)

* **Multi-dimensional dynamic model compression for efficient image super-resolution (WACV2022)** [[pdf]](https://openaccess.thecvf.com/content/WACV2022/papers/Hou_Multi-Dimensional_Dynamic_Model_Compression_for_Efficient_Image_Super-Resolution_WACV_2022_paper.pdf)




### Spatial methods
Spatial methods exploit spatial redundancies, such as unimportant regions, to save computations

#### Spatial per-pixel
* **PerforatedCNNs: Acceleration through Elimination of Redundant Convolutions**  (NIPS2016) [[pdf]](https://papers.nips.cc/paper/6463-perforatedcnns-acceleration-through-elimination-of-redundant-convolutions.pdf) [[matconvnet]](https://github.com/mfigurnov/perforated-cnn-matconvnet) [[caffe]](https://github.com/mfigurnov/perforated-cnn-caffe)  
M. Figurnov, A. Ibraimova, D. P. Vetrov, and P. Kohli
* **Spatially Adaptive Computation Time for Residual Networks** (CVPR2017)  [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Figurnov_Spatially_Adaptive_Computation_CVPR_2017_paper.pdf) [[tensorflow]](https://github.com/mfigurnov/sact)  
Figurnov M, Collins MD, Zhu Y, Zhang L, Huang J, Vetrov D, Salakhutdinov R. 
* **Pixel-wise Attentional Gating for Parsimonious Pixel Labeling** (WACV2019) [[pdf]](https://arxiv.org/pdf/1805.01556) [[matconvnet]](https://github.com/aimerykong/Pixel-Attentional-Gating)  
S. Kong and C. Fowlkes
* **Boosting the Performance of CNN Accelerators with Dynamic Fine-Grained Channel Gating** (MICRO2019) [[pdf]](https://zhouyuan1119.github.io/papers/cgnet-micro2019.pdf) 
Weizhe Hua, Yuan Zhou, Christopher De Sa, Zhiru Zhang, and G. Edward Suh
* **Dynamic Convolutions: Exploiting Spatial Sparsity for Faster Inference** (CVPR2020) [[pdf]](https://arxiv.org/pdf/1912.03203.pdf) [[Pytorch]](https://github.com/thomasverelst/dynconv)  
T. Verelst and T. Tuytelaars, “Dynamic Convolutions: Exploiting Spatial Sparsity for Faster Inference
* **Spatially Adaptive Inference with Stochastic Feature Sampling and Interpolation** (ECCV2020) [[pdf]](https://arxiv.org/pdf/2003.08866) [[pytorch]](https://github.com/zdaxie/SpatiallyAdaptiveInference-Detection)  
Z. Xie, Z. Zhang, X. Zhu, G. Huang, and S. Lin
* **Precision Gating: Improving Neural Network Efficiency with Dynamic Dual-Precision Activation** (ICLR2020) [[pdf]](https://arxiv.org/pdf/2002.07136) [[pytorch]](https://github.com/cornell-zhang/dnn-gating)  
Zhang Y, Zhao R, Hua W, Xu N, Suh GE, Zhang Z.

* **Dynamic Dual Gating Neural Networks** (ICCV2021) [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Dynamic_Dual_Gating_Neural_Networks_ICCV_2021_paper.pdf)  

* **Skip-Convolutions for Efficient Video Processing** (CVPR2021) [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Habibian_Skip-Convolutions_for_Efficient_Video_Processing_CVPR_2021_paper.pdf) [[pytorch]](https://github.com/Qualcomm-AI-research/Skip-Conv)  
[VID]

**Focal Sparse Convolutional Networks for 3D Object Detection** (CVPR2022) [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Focal_Sparse_Convolutional_Networks_for_3D_Object_Detection_CVPR_2022_paper.pdf)

#### Spatial per-block
* **SBNet: Sparse Blocks Network for Fast Inference** (CVPR2018) [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ren_SBNet_Sparse_Blocks_CVPR_2018_paper.pdf) [[tensorflow]](https://github.com/uber-research/sbnet)  
M. Ren, A. Pokrovsky, B. Yang, and R. Urtasun

* **Uncertainty based model selection for fast semantic segmentation** (MVA2019) [[pdf]](https://homes.esat.kuleuven.be/~konijn/publications/2019/Huang-MVA.pdf)

* **SegBlocks: Block-Based Dynamic Resolution Networks for Real-Time Segmentation** (ECCV2020 Workshop) [[pdf]](https://homes.esat.kuleuven.be/~konijn/publications/2020/verelst3.pdf) 
Thomas Verelst and Tinne Tuytelaars

* **Spatially Adaptive Feature Refinement for Efficient Inference** [[pdf]](https://ieeexplore.ieee.org/abstract/document/9609974/)  
Y Han, G Huang, S Song, L Yang, Y Zhang, H Jiang

* **BlockCopy: High-Resolution Video Processing with Block-Sparse Feature Propagation and Online Policies** (ICCV 2021) [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Verelst_BlockCopy_High-Resolution_Video_Processing_With_Block-Sparse_Feature_Propagation_and_Online_ICCV_2021_paper.pdf)
Thomas Verelst and Tinne Tuytelaars  
[VID]

#### Spatial warping
* **Learning to Zoom: A Saliency-Based Sampling Layer for Neural Networks** (ECCV2018) [[pdf]] [[pytorch]](https://github.com/recasens/Saliency-Sampler)
Adria Recasens, Petr Kellnhofer, Simon Stent, Wojciech Matusik, Antonio Torralba


#### Glances and dynamic crops
Takes crops to further refine predictions

* **Action Recognition using Visual Attention** (ICLR 2016 Workshop) [[pdf]](https://arxiv.org/pdf/1511.04119) [[theano]](https://github.com/kracwarlock/action-recognition-visual-attention)  
S. Sharma, R. Kiros, and R. Salakhutdinov

* **Recurrent Models of Visual Attention** (NIPS2014) [[pdf]](https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)  
V. Mnih, N. Heess, A. Graves, and  koray kavukcuoglu
* **Dynamic Capacity Networks** (ICML2016) [[pdf]](http://www.jmlr.org/proceedings/papers/v48/almahairi16.pdf) [[tensorflow]](https://github.com/beopst/dcn.tf) [[unofficial pytorch]](https://github.com/philqc/Dynamic-Capacity-Network-Pytorch)  
A. Almahairi, N. Ballas, T. Cooijmans, Y. Zheng, H. Larochelle, and A. Courville 
* **Glance and Focus: a Dynamic Approach to Reducing Spatial Redundancy in Image Classification** [[pdf]](https://arxiv.org/pdf/2010.05300)  
 Y. Wang, K. Lv, R. Huang, S. Song, L. Yang, and G. Huang
* **Learning Where to Focus for Eﬃcient Video Object Detection** (ECCV2020)  [[pdf]](https://arxiv.org/pdf/1911.05253.pdf) [[github]](https://github.com/jiangzhengkai/LSTS)  
Z. Jiang et al.

* **Adaptive Focus for Efficient Video Recognition** (2021) [[pdf]](https://arxiv.org/pdf/2105.03245)  
Yulin Wang, Zhaoxi Chen, Haojun Jiang, Shiji Song, Yizeng Han, Gao Huang  
[VID]

* **Adafocus v2: End-to-end training of spatial dynamic networks for video recognition** (2021) [[pdf]](https://arxiv.org/abs/2112.14238) [VID]

#### Other (dilation etc)
* **D^2Conv3D: Dynamic Dilated Convolutions for Object Segmentation in Videos**
[[pdf]](https://arxiv.org/abs/2111.07774)
Christian Schmidt, Ali Athar, Sabarinath Mahadevan, Bastian Leibe  
[VID]


### Adaptive resolution methods
Adaptive resolution methods adapt the processing resolution to the input image.

* **Resolution Adaptive Networks for Efficient Inference** (CPVR2020) [[pdf]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Resolution_Adaptive_Networks_for_Efficient_Inference_CVPR_2020_paper.pdf) [[pytorch]](https://github.com/yangle15/RANet-pytorch)  
 L. Yang, Y. Han, X. Chen, S. Song, J. Dai, and G. Huang
* **Resolution Switchable Networks for Runtime Eﬃcient Image Recognition** (ECCV2020) [[pdf]](https://arxiv.org/pdf/2007.09558) [[pytorch]](https://github.com/yikaiw/RS-Nets)  
Y. Wang, F. Sun, D. Li, and A. Yao
* **Dynamic Resolution Network** (2021) [[pdf]](https://arxiv.org/pdf/2106.02898)
* **Multi-dimensional dynamic model compression for efficient image super-resolution (WACV2022)** [[pdf]](https://openaccess.thecvf.com/content/WACV2022/papers/Hou_Multi-Dimensional_Dynamic_Model_Compression_for_Efficient_Image_Super-Resolution_WACV_2022_paper.pdf)


### Transformers
* **Dynamically Pruning Segformer for Efficient Semantic Segmentation** (Arxiv2021) [[pdf]](https://arxiv.org/pdf/2111.09499.pdf)  
Haoli Bai, Hongda Mao, Dinesh Nair

* **Spatio-Temporal Gated Transformers for Efficient Video Processing** (2021) [[pdf]](https://ml4ad.github.io/files/papers2021/Spatio-Temporal%20Gated%20Transformers%20for%20Efficient%20Video%20Processing.pdf)  
Yawei Li, Babak Ehteshami Bejnordi, Bert Moons, Tijmen Blankevoort, Amirhossein Habibian, Radu Timofte, Luc Van Gool  
[VID]

* **Not All Images are Worth 16x16 Words: Dynamic Transformers for Efficient Image Recognition** (NIPS2021) [[pdf]](https://papers.nips.cc/paper/2021/file/64517d8435994992e682b3e4aa0a0661-Paper.pdf)  
Yulin Wang, Rui Huang, Shiji Song, Zeyi Huang, Gao Huang

* **Multi-Exit Vision Transformer for Dynamic Inference**  (2021) [[pdf]](https://arxiv.org/pdf/2106.15183)  
A Bakhtiarnia, Q Zhang, A Iosifidis

* **Dynamic Grained Encoder for Vision Transformers** (NIPS2021) [[pdf]](https://papers.nips.cc/paper/2021/file/2d969e2cee8cfa07ce7ca0bb13c7a36d-Paper.pdf)  
Song, Lin, Songyang Zhang, Songtao Liu, Zeming Li, Xuming He, Hongbin Sun, Jian Sun, and Nanning Zhen

* **A-ViT: Adaptive Tokens for Efficient Vision Transformer** (CVPR2022) [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/html/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.html)


## Dynamic filters/weights
* **Dynamic filter networks** (NIPS2016) [[pdf]](http://papers.nips.cc/paper/6578-improving-pac-exploration-using-the-median-of-means.pdf)  
Jia, X., De Brabandere, B., Tuytelaars, T., & Gool, L. V.

* **Dynamic region-aware convolution** (CVPR2021) [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Dynamic_Region-Aware_Convolution_CVPR_2021_paper.pdf)  
Chen, J., Wang, X., Guo, Z., Zhang, X., & Sun, J.

* **Decoupled Dynamic Filter Networks** (CVPR2021) [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Decoupled_Dynamic_Filter_Networks_CVPR_2021_paper.pdf)

* **Involution: Inverting the inherence of convolution for visual recognition** (CPVR2021) [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Involution_Inverting_the_Inherence_of_Convolution_for_Visual_Recognition_CVPR_2021_paper.pdf) [[pytorch]](https://github.com/d-li14/involution) [[unofficial tf]](https://github.com/ariG23498/involution-tf)

* **Adaptive Convolutions with Per-pixel Dynamic Filter Atom** (ICCV2021) [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Adaptive_Convolutions_With_Per-Pixel_Dynamic_Filter_Atom_ICCV_2021_paper.pdf)  

## Quantization
* **Instance-Aware Dynamic Neural Network Quantization** (CVPR2022) [[pdf]] (https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Instance-Aware_Dynamic_Neural_Network_Quantization_CVPR_2022_paper.html)

## Mixture of experts
* **HydraNets: Specialized Dynamic Architectures for Efficient Inference** (CVPR2019) [[pdf]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mullapudi_HydraNets_Specialized_Dynamic_CVPR_2018_paper.pdf)   
Teja Mullapudi R, Mark WR, Shazeer N, Fatahalian K.
* **Outrageously large neural networks: The sparsely-gated mixture-of-experts layer** (ICLR 2017) [[pdf]](https://arxiv.org/pdf/1701.06538.pdf%22%20%5Ct%20%22_blank)  [[unofficial pytorch]](https://github.com/davidmrau/mixture-of-experts)  
Shazeer N, Mirhoseini A, Maziarz K, Davis A, Le Q, Hinton G, Dean J. 



### Other

#### Video
(if not listed above with [VID] tag)
* **Leaky Gated Cross-Attention for Weakly Supervised Multi-Modal Temporal Action Localization** (WACV2022) [[pdf]](https://openaccess.thecvf.com/content/WACV2022/html/Lee_Leaky_Gated_Cross-Attention_for_Weakly_Supervised_Multi-Modal_Temporal_Action_Localization_WACV_2022_paper.html) [VID]

* **ELIχR: Eliminating Computation Redundancy in CNN-Based Video Processing** (RSDHA2021) [[IEEE]](https://ieeexplore.ieee.org/document/9651154) [ VID]