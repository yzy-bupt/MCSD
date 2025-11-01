---
layout: project_page
permalink: /

title: "Multi-View Captioning with Semantic Delta Re-Ranking for Zero-Shot Composed Video Retrieval"
authors:
    Zhixiang Ding<sup><span style="color:blue">1</span></sup>, Lilong Liu<sup><span style="color:green">2</span></sup>, Zhenyu Yang<sup><span style="color:blue">1</span></sup>, Shengsheng Qian<sup><span style="color:blue">1</span></sup>
affiliations:
    <sup><span style="color:blue">1</span></sup>Institute of Automation, Chinese Academy of Sciences, <sup><span style="color:green">2</span></sup>Zhengzhou University
conference:
    <strong><span style="color:gray">ICIG'2025</span> (<span style="color:red">Oral</span>🔥)</strong>
paper: https://icig.csig.org.cn/2025/6172/list.html
code: https://github.com/yzy-bupt/MCSD
---

<!-- Using HTML to center the abstract -->

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
Composed Video Retrieval (CVR) aims to retrieve video relevant to a query video while incorporating specific changes described in modification text. For Zero-Shot Composed Video Retrieval (ZS-CVR), current methods utilize vision-language models to convert the query video into a single caption, subsequently merged with modification text to generate an edited caption for retrieval. However, the modification text doesn't clearly specify which elements to preserve from the query video, leading to possible misalignment between edited caption and target video. Additionally, the final retrieval result should not be determined solely by the similarity between edited caption and candidate videos but also incorporate the semantic delta arising from the modification text. To address these issues, we propose Multi-View Captioning with Semantic Delta Re-Ranking (MCSD) method for ZS-CVR. Specifically, the Multi-View Captioning Module to generate captions covering potential semantics of the target video, the Semantic Delta Re-Ranking Module that computes the semantic delta between the original and edited captions, to adjust similarity scores and re-ranks the retrieval results. Extensive experiments on two benchmarks demonstrate that the proposed MCSD method achieves state-of-the-art performance in ZS-CVR.
        </div>
    </div>
</div>


---

## **Demo**
<video autoplay controls muted loop playsinline height="100%">
  <source src="/static/video/Demo.mp4" type="video/mp4">
</video>

> This is a data demo in our MCSD.

## **Overview**
**Motivation: _Video content is inherently dense in semantic information._** A single caption often fails to capture the full semantics of a target video, whereas captions generated from multiple perspectives can provide more comprehensive coverage of its potential meanings.

![overview](/static/image/case.png)

*Figure 1: Illustration of multiple perspectives vs. single caption.*



## **MCSD**



To address these issues, we propose Multi-View Captioning with Semantic Delta Re-Ranking (MCSD) for ZS-CVR. Our method features:

(1) **Multi-View Captioning Module** to generate captions covering potential semantics of the target video;

(2) **Semantic Delta Re-Ranking Module** that computes the semantic delta between original and edited captions to adjust similarity scores and re-rank retrieval results.

![model_framework](/static/image/method.png)

*Figure 2: Architecture of the proposed MCSD method.*



## Citation
```
TBC
```
