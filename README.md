# detection_ai
Project which help people to recognize generated text.

Laubch steps:

1) Launch the parser on kaggle (https://www.kaggle.com/code/statiana/parser-main, don't forget open the last version) and get the files that need to be read. An example output file is "parsed_file".
2) Before starting, you need to clear all received files using "processing_files.py", you should put directory with files from previous step. We will delete those that are known to be broken. 
3) Using "processing_text.py", we read all PDFs into one large json using modules from "paper_parse.py" and perform filtering. You should put directory with files from previous step.
4) Using "dim_script.py" we calculate the dimensions of PHdim and MLE. However, generated texts have already been added here, which you need to do yourself using the openAI API. You must use a dataset ("gen_human_papers.json" in code) with generated texts and real texts, dataset is built according to the Kaggle code (https://www.kaggle.com/code/statiana/parse-gen-text-new). As an example of the output file, there is the file "Example_dim". We save it in separate json files for quick pre-processing.
5) Using "for_diploma_graph.py" and "mle_stats.py", we draw graphs and train models for both methods of calculating the internal dimension.

The "PHD&MLE.ipynb" code works only with a ready-made dataset and calculates metrics only for it.

Some examples were also posted here https://www.kaggle.com/datasets/statiana/dimgeneratedhuman-texts.
