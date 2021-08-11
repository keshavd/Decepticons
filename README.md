<h1><img src="decepticons.png"
  width="32"
  height="32"
  style="float:left;">&ensp;Decepticons</h1>
Extending HuggingFace's `transformers` package with new models.

## Description
Many of HuggingFace's finetuning models have redundant code. 
These patterns have been refactored into `Mixins` and new 
generalized training `Heads` allowing for more flexibility.

## New Models

* BERT
    * BertForSequenceClassification
    * BertForTokenClassification


* RoBERTA
    * RobertaForSequenceClassification
    * RobertaForTokenClassification

## To Be Added:
* BERT
    * BertForTokenCrfClassification
    