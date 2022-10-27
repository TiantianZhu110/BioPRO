# BIOPRO 
Code for IJCAI 2022 paper: **Enhancing Entity Representations with Prompt Learning for Biomedical Entity Linking.**

# Introduction
We propose a two-stage entity linking algorithm to enhance the entity representations based on prompt learning. The first stage includes a coarser-grained retrieval from a representation space defined by a bi encoder that independently embeds the mentions and entities’ surface forms. Unlike previous one-model-fits-all systems, each candidate is then re-ranked with a finer-grained encoder based on prompt-tuning that concatenates the mention context and entity information. Extensive experiments show that our model achieves promising performance improvements compared with several state of-the-art techniques on the largest biomedical public dataset MedMentions and the NCBI disease corpus.
We also observe by cases that the proposed prompt-tuning strategy is effective in solving both the variety and ambiguity challenges in the linking task. 

<div align='center'>
<img src="./arc.pdf?version=15&modificationDate=1596786732179&api=v2"/>
</div>

#Requirement
--
```
python: 3.8
PyTorch: 1.9.0
transformers: 4.10.0
openprompt: 0.1.1
```
you can go [here](https://github.com/thunlp/OpenPrompt) to know more about OpenPrompt. 


# How to run the code?

1.Download the pytorch based pubmed bert pretrained model from [here](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/tree/main), and put it to the folder "pretrain".

2.Generate training sample and test sample data according to the file data_process/data\_process.py.Provide entity dictionary file, entity type information file, and corresponding sample files of mention and gold entity according to the code description to generate corresponding training samples.

3.Run prompt\_ranking/prompt\_medicine\_train.py to train the model.

4.Run prompt\_ranking/prompt\_medicine\_predict.py to predict the result.

5.Run prompt\_retrieval/prompt\_entity\_vector.py to generate mention and entity vector with prompt model.

6.Run prompt\_retrieval/vector\_search.py to serach top N candidates with prompt model.


