# Spell Correction  
In spell correction tasks, inputs are sentences which includes errors in 4 category and outputs should be the correct sentence which recognize error and fix it properly.  
In this part, the extracted data are used in our model.  
we picked a transformer model for this task :  
  ![transformer structure](https://miro.medium.com/v2/resize:fit:1400/1*10K7SmGoJ5zAtjkGfNfjkg.png)  
This section includes three stages:
## Preprocess 
To preprocessing the data, additional character such as @,#,url,... and any noises will be removed.  
In addition, a word_tokenizer is used with hazm lib. It devide sentences to their words as tokens.  
## Train 
80% of data will be picked randomly as train dataset. After that 20% of train dataset will be split randomly as validation dataset to avoid overfitting.  
Input data will be batched in 64 size. This means in 9200 data at all ( train and test ) each epoch has 77 iteratinos.  
The model has been trained in 30 epoches.  
## Test 
After training stage, model is ready to testing.  
Model is tested with 5 scales:  
- accuracy
- bleu score
- f1-score
- precision
- recall
