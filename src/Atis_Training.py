from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from transformer_chat import TransformerChatbot
import pandas as pd
import random

# Loading atis-datasets
raw_dataset = load_dataset("tuetschek/atis", split="train")

# Loading tokenizer from file
tokenizer = Tokenizer.from_file('tokenizer.json')

# Create synthetic responses for ATIS queries for training purposes
def create_response_for_intent(intent, text):
    """Create synthetic responses for ATIS intents"""
    responses = {
        'atis_flight': [
            "I can help you with flight information. What specific details do you need?",
            "I'll search for flights matching your criteria. Please provide departure and arrival cities.",
            "Let me find available flights for you. When would you like to travel?"
        ],
        'atis_flight_no': [
            "I can help you with flight number information. Please provide the flight number.",
            "Let me search for details about that flight number.",
            "I'll look up information for that specific flight."
        ],
        'atis_airfare': [
            "I can help you find airfare information. What's your travel route?",
            "Let me search for the best airfare options for your trip.",
            "I'll check current airfare prices for your destination."
        ],
        'atis_airline': [
            "I can help you with airline information. Which airline are you looking for?",
            "Let me provide information about that airline.",
            "I'll search for details about the airline you mentioned."
        ],
        'atis_abbreviation': [
            "I can help you with airport abbreviations. Which abbreviation do you need?",
            "Let me explain that airport abbreviation for you.",
            "I'll provide the full name for that airport code."
        ],
        'atis_airport': [
            "I can help you with airport information. Which airport are you looking for?",
            "Let me provide details about that airport.",
            "I'll search for information about the airport you mentioned."
        ],
        'atis_distance': [
            "I can help you calculate distances between airports. Which airports are you interested in?",
            "Let me calculate the distance for you.",
            "I'll provide distance information between those locations."
        ],
        'atis_ground_service': [
            "I can help you with ground transportation services. What type of service do you need?",
            "Let me find ground transportation options for you.",
            "I'll search for available ground services at your destination."
        ],
        'atis_aircraft': [
            "I can help you with aircraft information. What type of aircraft are you looking for?",
            "Let me provide details about that aircraft type.",
            "I'll search for information about the aircraft you mentioned."
        ],
        'atis_capacity': [
            "I can help you with capacity information. What specific capacity details do you need?",
            "Let me check the capacity for that flight or aircraft.",
            "I'll provide capacity information for your query."
        ],
        'atis_quantity': [
            "I can help you with quantity information. What specific quantity are you looking for?",
            "Let me check the quantity for that item or service.",
            "I'll provide quantity information for your request."
        ],
        'atis_meal': [
            "I can help you with meal information. What type of meal service are you looking for?",
            "Let me check meal options for your flight.",
            "I'll provide information about meal services available."
        ],
        'atis_cheapest': [
            "I can help you find the cheapest options. What's your travel route?",
            "Let me search for the most affordable options for your trip.",
            "I'll find the cheapest flights or services for you."
        ],
        'atis_restriction': [
            "I can help you with travel restrictions. What type of restrictions are you asking about?",
            "Let me check the restrictions for your travel plans.",
            "I'll provide information about travel restrictions."
        ],
        'atis_day_name': [
            "I can help you with day information. What specific day are you looking for?",
            "Let me check the schedule for that day.",
            "I'll provide information about flights or services on that day."
        ]
    }
    
    # Get base responses for the intent calssification datasets
    base_responses = responses.get(intent, [
        "I can help you with that. Please provide more details.",
        "Let me assist you with your request.",
        "I'll help you find the information you need."
    ])
    
    # For variety
    if "flight" in text.lower():
        base_responses.extend([
            "I can help you book a flight. What are your travel dates?",
            "Let me search for available flights for you.",
            "I'll help you find the best flight options."
        ])
    
    return random.choice(base_responses)

# Create training data with question-answer pairs
def create_training_pairs():
    training_data = []
    
    for item in raw_dataset:
        question = item['text']
        intent = item['intent']
        response = create_response_for_intent(intent, question)
        
        # Tokenize question and response
        question_encoding = tokenizer.encode(question)
        response_encoding = tokenizer.encode(response)
        
        # Add the specially defined tokens
        question_ids = [tokenizer.token_to_id("[CLS]")] + question_encoding.ids + [tokenizer.token_to_id("[SEP]")]
        response_ids = [tokenizer.token_to_id("[CLS]")] + response_encoding.ids + [tokenizer.token_to_id("[SEP]")]
        
        training_data.append({
            'question_ids': question_ids,
            'response_ids': response_ids,
            'question_len': len(question_ids),
            'response_len': len(response_ids)
        })
    
    return training_data

# Create custom dataset for training
class AtisGenerationDataset(Dataset):
    def __init__(self, training_data, tokenizer, max_length=128):
        self.training_data = training_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        item = self.training_data[idx]
        
        # Pad sequences
        question_ids = item['question_ids'][:self.max_length//2]
        response_ids = item['response_ids'][:self.max_length//2]
        
        # Pad with PAD token
        question_ids += [tokenizer.token_to_id("[PAD]")] * (self.max_length//2 - len(question_ids))
        response_ids += [tokenizer.token_to_id("[PAD]")] * (self.max_length//2 - len(response_ids))
        
        return (
            torch.tensor(question_ids),
            torch.tensor(response_ids),
            torch.tensor(item['question_len']),
            torch.tensor(item['response_len'])
        )

# Create training data
print("Creating training data...")
training_data = create_training_pairs()
print(f"Created {len(training_data)} training pairs")

# Prepare DataLoader
atis_dataset = AtisGenerationDataset(training_data, tokenizer)
dataloader = DataLoader(atis_dataset, batch_size=16, shuffle=True)

# Prepare model with all the neccessary parameters
vocab_size = tokenizer.get_vocab_size()
model = TransformerChatbot(
    vocab_size=vocab_size,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_roles=2,
    max_turns=16,
    num_slots=len(set(item['intent'] for item in raw_dataset)),
    dropout=0.1
)

# Using gpu - cuda for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop for generation
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

print("Starting training...")
for epoch in range(10):  # 10 epochs for fast training
    model.train()
    total_loss = 0
    for batch_idx, (question_ids, response_ids, question_lens, response_lens) in enumerate(dataloader):
        question_ids = question_ids.to(device)
        response_ids = response_ids.to(device)
        
        batch_size, seq_len = question_ids.shape
        
        # Dummy roles and turns
        roles = torch.zeros_like(question_ids)
        turns = torch.zeros_like(question_ids)
        # Forward pass
        gen_logits, slot_logits = model(
            question_ids, response_ids,
            roles, roles,  
            turns, turns   
        )
        
        # Calculate loss for generation (teacher forcing)
        target_ids = response_ids[:, 1:]  # Remove [CLS] token
        gen_logits = gen_logits[:, :-1, :]  # Remove last position        
        # Flatten for loss calculation
        gen_logits_flat = gen_logits.reshape(-1, vocab_size)
        target_ids_flat = target_ids.reshape(-1)
        loss = loss_fn(gen_logits_flat, target_ids_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    # Averaging the losses
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# Save model
print("Saving model...")
torch.save(model.state_dict(), 'atis_transformer.pt')
print("Training completed!")