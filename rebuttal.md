\begin{markdown}

# Review 2 6CJo rating3

We sincerely appreciate your time and efforts in reviewing our paper! Based on your review, we added a detailed discussion and additional experiments. 


**Q1: About the details on method: what diffusion time-step is used to extract TVP features during inference?**

% ANS: Thank you for your question! As mentioned in Section 5.1 ("Policy Rollout Details"), we extract TVP features from the first diffusion step during inference. This allows us to achieve a high control frequency. Additionally, we visualize the features from the first denoising step in Figure 4 of the paper, which confirms that it already contains substantial future information.

% Furthermore, we conducted an ablation study on the diffusion time-step, which is presented in Table 10 of the original paper (In Appendix). The results show that features from the second denoising step perform similarly to those from the first step. Therefore, we use the features from the first step to achieve the highest possible frequency, which ranges from 7 to 10 Hz.

Thank you for your question. In our pipeline, we set the SVD noise scale and diffusion timestep to 20. We conducted preliminary experiments with diffusion timesteps of 10, 20, and 30 and found that 20 yielded slightly better performance. However, the overall performance of VPP is not sensitive to the choice of timestep. The results for different timesteps are shown below:

|   CALVIN ABC-D       | Avg.Len|
| :-- | -- |
|    VPP time-step 10  | 4.21  |
|    VPP time-step 20  | 4.33  |
|    VPP time-step 30  | 4.25 |

---

**Q2: In Table 4, do you have results with SVD pretrain, but only internet data training? I.e. the impact of SVD pre-training for the robotics tasks.**

ANS: Thank you for the insightful question! Following your suggestion, we fine-tuned the SVD model using only internet data, without incorporating downstream robotics data. As shown below, the results indicate a clear performance decline. We believe this is because **video prediction quality plays a crucial role in action learning**. In the VPP framework, fine-tuning the video model on robot datasets enhances video prediction quality within the specific domain, allowing it to better capture the dynamics of robotic data and potentially improve performance.



|               | Avg.Len|
| :-- | -- |
|    VPP w/o internet data    | 3.97  |
|    VPP w/o down-stream robot dataset (newly added)   | 3.31  |
|    VPP   | 4.33 |


---

**Q3: Please add table comparing against inference speed of baselines.**

ANS: Thank you for the constructive suggestion! Follow your suggestion, We add a comparison of the inference speed here. All inference time is evaluated on single NVIDIA 4090 GPU and average on 100 runs. We can notice that VPP achieves the best performance while keeping high frequency. Other  method containing video/imgae diffusion (e.g., Susie/Uni-Pi) requires long time to denoise a complete video.

|     CALVIN ABC-D  | inference Time|  Avg. Len | 
| :-- | -- |-- |
|    Diffusion policy      | ~100ms |  0.56 |
|    3d-diffuser actor     | ~600ms |  3.35 |
|    Susie    |  ~5100ms | 2.69 |
|    GR-1     |  ~90ms   | 3.06 |
|    MDT      |  ~110ms  | 1.55 |
|    Uni-Pi   |  ~5500ms  | 0.92 |
|    VPP      |  ~140ms  | 4.33|

---

Thank you again for your time and effort in reviewing our work! We hope our clarification can solve all your concerns, and we are always ready to answer any further questions!



\end{markdown}