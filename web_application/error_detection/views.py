# error_detection/views.py

from django.shortcuts import render
from transformers import AutoModel, AutoConfig ,AutoTokenizer
import torch
model_path='error_detection/models/Without_Nationality/won2_directory'
config = AutoConfig.from_pretrained(model_path)
model_won = AutoModel.from_pretrained(model_path, config=config, use_safetensors=True)

model_path='error_detection/models/Witho_Batch_Nationality/BNB'
config = AutoConfig.from_pretrained(model_path)
model_BNB = AutoModel.from_pretrained(model_path, config=config, use_safetensors=True)

model_path='error_detection/models/With_Nationality/NB2'
config = AutoConfig.from_pretrained(model_path)
model_NB = AutoModel.from_pretrained(model_path, config=config, use_safetensors=True)

loaded_tokenizer = AutoTokenizer.from_pretrained("error_detection/tokenizer")


def predict_sentence(sentence,model,tokenizer=loaded_tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, is_split_into_words=False)
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
    classification_head = torch.nn.Linear(last_hidden_state.shape[2], 1)  # Binary classification
    token_logits = classification_head(last_hidden_state)  # shape: (batch_size, seq_len, 1)
    token_logits = token_logits.squeeze(-1)  # shape: (batch_size, seq_len)
    token_probabilities = torch.sigmoid(token_logits)
    token_predictions = (token_probabilities > 0.5).int()
    return token_predictions[0].tolist()
def error_detection_view(request):

    text=[]
    batch_nationality = ['Arab','Latin','Turk','Pars','India','Russia','china','Other']
    if request.method == 'POST':
        text = []
        sentence = request.POST.get('sentence')
        nationality = request.POST.get('nationality')
        if nationality=="without_nationality":
            print("without")
            predict = predict_sentence(sentence, model_won)
            token = loaded_tokenizer(sentence)['input_ids']

            print(predict)
            print(token)
            for i in range(0, len(predict)):
                if predict[i] != 0:
                    print(token[i])
                    if token[i]!=2 and token[i]!=4:
                        text.append(loaded_tokenizer.convert_ids_to_tokens(token[i]))
        elif nationality in batch_nationality:
            sentence = "[" + nationality + "] " + sentence
            predict = predict_sentence(sentence, model_BNB)
            token = loaded_tokenizer(sentence)['input_ids']
            for i in range(0, len(predict)):
                if predict[i] != 0:
                    print(token[i])
                    if token[i]!=2 and token[i]!=4 and token[i]!=24 and token[i]!=26 and loaded_tokenizer.convert_ids_to_tokens(token[i])!=nationality:
                        text.append(loaded_tokenizer.convert_ids_to_tokens(token[i]))
        else:
            sentence = "[" + nationality + "] " + sentence
            predict = predict_sentence(sentence, model_NB)
            token = loaded_tokenizer(sentence)['input_ids']
            print(token)
            for i in range(0, len(predict)):
                if predict[i] != 0:
                    print(token[i])
                    if token[i]!=2 and token[i]!=4 and token[i]!=24 and token[i]!=26 and loaded_tokenizer.convert_ids_to_tokens(token[i])!=nationality:
                        text.append(loaded_tokenizer.convert_ids_to_tokens(token[i]))



    return render(request, 'error_detection.html', {'corrected_sentence': str(text)})
