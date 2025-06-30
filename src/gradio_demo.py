import torch
import gradio as gr
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from transformer_chat import TransformerChatbot

# Load tokenizer & wrap for HF API
tokenizer_obj = Tokenizer.from_file("tokenizer.json")
hf_tok = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_obj,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerChatbot(
    vocab_size=hf_tok.vocab_size,
    d_model=512, num_heads=8, d_ff=2048,
    num_encoder_layers=6, num_decoder_layers=6,
    num_roles=2, max_turns=16, num_slots=22,
    dropout=0.1
).to(device)
model.load_state_dict(torch.load("atis_transformer.pt", map_location=device))
model.eval()

# Generation function
def chat_fn(prompt):
    # Encode user input
    enc = hf_tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    src_ids  = enc.input_ids.to(device)
    # For cross-attention, we don't need to mask the encoder output
    src_mask = None

    # Roles & turns (user=0)
    roles = torch.zeros_like(src_ids)
    turns = torch.zeros_like(src_ids)

    # Encode
    with torch.no_grad():
        enc_out = model.encode(src_ids, roles, turns, src_mask)

        # Generate reply token-by-token
        cls_id = hf_tok.cls_token_id
        sep_id = hf_tok.sep_token_id
        dec_input = torch.tensor([[cls_id]], device=device)
        dec_roles = torch.zeros_like(dec_input)
        dec_turns = torch.zeros_like(dec_input)

        generated = []
        for step in range(50):
            T = dec_input.size(1)
            # Create causal mask for decoder (upper triangular = masked)
            # PyTorch's MultiheadAttention expects a 2D mask where True = masked
            causal_mask = torch.triu(torch.ones((T, T), device=device), diagonal=1).bool()
            tgt_mask = causal_mask

            logits = model.decode(dec_input, enc_out, dec_roles, dec_turns, src_mask, tgt_mask)
            
            # Get the last token's logits
            last_logits = logits[0, -1, :]
            
            # Apply repetition penalty
            if generated:
                for token_id in set(generated):
                    last_logits[token_id] *= 0.7  # Penalize repeated tokens
            
            # Sample with temperature instead of greedy decoding
            temperature = 0.8
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
            
            # Debug: print the token being generated
            token_text = hf_tok.decode([next_id.item()])
            print(f"Step {step}: Generated token ID {next_id.item()} -> '{token_text}'")
            
            if next_id.item() == sep_id:
                print("Found SEP token, stopping generation")
                break
                
            generated.append(next_id.item())
            dec_input = torch.cat([dec_input, next_id.unsqueeze(0)], dim=1)
            dec_roles = torch.cat([dec_roles, torch.zeros_like(next_id).unsqueeze(0)], dim=1)
            dec_turns = torch.cat([dec_turns, torch.zeros_like(next_id).unsqueeze(0)], dim=1)
            
            # Early stopping if we're stuck in a loop
            if len(generated) >= 3 and len(set(generated[-3:])) == 1:
                print("Detected repetition loop, stopping generation")
                break

        output_ids = [cls_id] + generated + [sep_id]
        reply = hf_tok.decode(output_ids, skip_special_tokens=True)

    return reply

# Build Gradio interface
interface = gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Transformer Chatbot Demo (currently trained with ATIS dataset)",
    description="Ask flight-related questions and get an answer."
)

if __name__ == "__main__":
    interface.launch(share=True)
