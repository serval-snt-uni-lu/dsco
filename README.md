# DSCo: Domain Series Corpus

DSCo is a language modeling based approach towards time series classification.
This repository is to support our paper titled "DSCo: A Language Modeling Approach for Time Series Classification", which is submitted to MLDM.

## How DSCo works

![Flowchart of training phase](fig/flowchart.png)

![Flowchart of classification phase](fig/flowchart-classification.png)

* Convert real-valued time series into texts using SAX (both training and testing data)
* Build per-class corpus from texts (model building)
* Calculate fitness scores (using the text segmentation techniques)
* Assign classification labels based on segmentation fitness scores

## Performance comparison between SAX-distance based 1NN and DSCo

![Comparison of SAX and DSCo classification accuracy](fig/sax-dsco.png)

As shown, DSCo (red lines) performs better in the majority of datasets from UCR.
Numbers in the plots indicate AUC for SAX and DSCo respectively (bigger AUC indicate higher overall accuracy).

## Performance comparison between DTW-distance based 1NN and DSCo (best results for both)

![Comparison of DTW and DSCo classification accuracy](fig/dtw-dsco.png)
