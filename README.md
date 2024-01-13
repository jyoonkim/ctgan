# ctGAN: Combined Transformation of Gene Expression and Survival Data with Generative Adversarial Network

# Abstract

Recent studies have extensively used deep learning algorithms to analyze gene expression to predict disease diagnosis, treatment effectiveness, and survival outcomes. Survival analysis studies on diseases with high mortality rates, such as cancer, are indispensable. However, deep learning models are plagued by overfitting owing to the limited sample size relative to the large number of genes. Consequently, the latest style-transfer deep generative models have been implemented to generate gene expression data. However, these models are limited in their applicability for clinical purposes because they generate only transcriptomic data. Therefore, this study proposes ctGAN, which enables the combined transformation of gene expression and survival data using a generative adversarial network (GAN). ctGAN improves survival analysis by augmenting data through style transformations between breast cancer and 11 other cancer types. We evaluated the concordance index (C-index) enhancements compared with previous models to demonstrate its superiority. Performance improvements were observed in in 9 of the 11 cancer types. Moreover, ctGAN outperformed previous models in 8 out of the 11 cancer types, with colon adenocarcinoma (COAD) exhibiting the most significant improvement (median C-index increase of approximately 14.8%). Furthermore, integrating the generated COAD enhanced the log-rank p-value (0.041) compared with using only the real COAD (p-value = 0.797). Based on the data distribution, we demonstrated that the model generated highly plausible data. In clustering evaluation, ctGAN exhibited the highest performance in most cases (89.62%). These findings suggest that ctGAN can be meaningfully utilized to predict disease progression and select personalized treatments in the medical field.

# Data

The data that support the findings of this study are openly available at the following URL/DOI: www.cancer.gov/tcga
We downloaded raw gene expression, survival data, clinical information, ensemble to gene symbol data using tcga_data_download.R
Due to periodic updates, downloaded data may vary depending on the timing (We downloaded it in July 2022).

# Codes

Test the code using the main.py file and check the results for COAD
