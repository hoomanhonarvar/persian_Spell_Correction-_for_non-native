# persian_Spell_Correction_for_non-native
Design and implementation of an applicatiojn to detect Persian spelling  mistakes of non-native language learners by neural network.

Learning to write in a foreign language often presents unique challenges, especially when it comes to spelling. Non-native learners of Persian (Farsi) frequently make spelling mistakes influenced by their native languages. Understanding the patterns behind these errors can provide valuable insights into the language acquisition process and improve automated spell correction systems.

The goal of this project is to investigate the relationship between a learner's nationality and the types of spelling errors they make when writing in Persian. By exploring this connection, we aim to develop a more context-aware spell correction model that not only corrects errors but also leverages the learner’s nationality to better predict the types of mistakes they are likely to make. This approach can lead to a more personalized and accurate correction system, tailored to the specific challenges faced by learners from different linguistic backgrounds.

Our spell correction model is designed using a sequence-to-sequence transformer architecture, where the primary task is correcting spelling mistakes and the auxiliary task involves predicting the nationality of the writer. This dual-task approach allows the model to integrate contextual information about the learner's background, resulting in more informed and effective corrections.

# Model
Project has two essential parts which are spell correction and Error detection models.  
both models has been trained and tested with same data but in diferent hyperparameter and structure.  
## Spell Correction
Spell correction is a crucial task in natural language processing (NLP) aimed at identifying and correcting spelling errors in written text. The importance of this task stems from the fact that human-written text is often prone to errors due to typos, phonetic similarities, or lack of knowledge about correct spelling. Effective spell correction systems can significantly enhance the readability and quality of text, especially in applications like search engines, chatbots, and text processing tools.

In this project, we implement a transformer-based spell correction model designed to learn and correct spelling errors in a sequence-to-sequence framework. The system is trained on pairs of incorrect and correct sentences, allowing it to automatically predict the correct spelling based on input sentences. Additionally, the model is extended with an auxiliary task of predicting the nationality of the writer, offering richer contextual understanding of the input text.
## Error Detection
Error detection is a fundamental task in natural language processing (NLP) that focuses on identifying incorrect usages of grammar, punctuation, and word choice in a given text. For languages like Persian, where automated tools are limited, building an efficient error detection system can significantly enhance applications such as grammar checking, text correction, and language learning.

In this project, we leverage **ParsBERT**, a state-of-the-art pre-trained language model specifically designed for the Persian language, to detect errors in Persian texts. ParsBERT is based on the BERT architecture, which has been fine-tuned on a large corpus of Persian texts, making it highly effective for a range of NLP tasks including error detection. By utilizing this model, we aim to build a robust system that can analyze Persian sentences and flag grammatical, syntactic, or spelling errors with high accuracy.

The goal of this error detection model is to improve the quality of Persian text by automatically identifying and highlighting areas that need correction, helping both native speakers and language learners alike.

# Result
the result of both models are illustrated as below:  
## Spell Correction  
|         | Accuracy | Bleu Score | f1-Score | Precision | Recall |
|---------|----------|------------|----------|-----------|--------|
|without_nationality| 0.71 | 0.59 | 0.70 | 0.74 | 0.71 |
|with_nationality| 0.65 | 0.55 | 0.64 | 0.68 | 0.65|
|with_batch_nationality| 0.70 | 0.58 | 0.69 | 0.73 | 0.70 |



## Error Detection

|          | Accuracy | f1-Score|
|----------|----------|---------|
| without_nationality| 0.51 | 0.60 |
| with_nationality| 0.49 | 0.54 |
| with_batch_nationality| 0.50 | 0.61 |

