import spacy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load spaCy model for information extraction
nlp = spacy.load('en_core_web_sm')

def extract_details(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Extract named entities, verbs, and noun chunks from the text
    entities = [ent.text for ent in doc.ents]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    return entities + verbs + noun_chunks

def generate_text(prompt):
    # Encode the prompt to input IDs and pass it to the model
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100)

    # Decode the output IDs to words
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text

# Extract details from a text
text = "Apple Inc. is planning to open a new store in San Francisco."
details = extract_details(text)

# Generate a paragraph about the extracted details
prompt = ' '.join(details)
paragraph = generate_text(prompt)

print(paragraph)
