# clinical-notes-vectorization

This is a comparative study of different document vectorization methods on real-world clinical notes to encode medical information. In this task, three different vectorization methods are studied and three different downstream use cases are proposed and evaluated.

Each step of my project progression is presented as a jupyter notebook to better illustrate my thought process. You can also find a slide deck to get an overview of my project below: 
https://docs.google.com/presentation/d/1ARlCjmF2d3Ciec9u25OiypVh2nILu5S31yitGSotnmM/edit?usp=sharing

Below is a brief description about the project files and folders:
```
├── data (directory for initially provided data)
├── processed_data (directory for processed clinical notes, vectors and vocab files)
├── src
│   ├── evaluate.py (Functions for evaluating document vectors)
│   ├── plotutils.py (Functions for plotting and visualization)
│   └── preprocess.py (Functions for preprocess text and vectorization)
├── 1_Data_Preprocess.ipynb (Inital data understanding and preprocessing)
├── 2_Exploratory_Data_Analysis.ipynb (Exploratory Data Analysis and further text normalization)
├── 3_Vectorization_Tfidf.ipynb (TF-IDF Vectorization method)
├── 4_Vectorization_Word_Vector_Aggregation.ipynb (Word Vector Aggregation Vectorization method)
├── 5_Vectorization_ClinicalBert.ipynb (Neural Language Model Vectorization method)
├── 6_Classification_Task.ipynb  (Use case as a classification task)
├── 7_Similarity_Ranking_Task.ipynb (Use case as a similarity ranking task)
└── 8_Clustering_Task.ipynb (Use case as a clustering task)
```
