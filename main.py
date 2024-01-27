import gradio as gr
from transformers import AutoTokenizer, AutoModel
import pickle

# Load the DarijaBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
model = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT")

# Load the classifier
pac = pickle.load(open('model/pac.sav', 'rb'))

def predict(text):
    # Tokenize the text
    encoded_input = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors='pt')

    # Perform inference
    output = model(**encoded_input)

    # Get the sentence embedding
    sentence_embedding = output.last_hidden_state.detach().numpy()

    # Reshape the embedding
    reshaped_embedding = sentence_embedding.reshape(1, -1)

    # Perform prediction
    prediction = pac.predict(reshaped_embedding)

    # Return the prediction
    return "Positive" if prediction[0] == 0 else "Negative"

# Gradio app
iface = gr.Interface(fn=predict, inputs="text", outputs="text")
iface.launch()
