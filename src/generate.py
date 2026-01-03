import torch

def generate_text(model, token_ids, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        input_ids = token_ids[:, -context_size:]      # input text that the model is allowed to see

        with torch.no_grad():
            logits = model(input_ids)                

        logits = logits[:, -1, :]                       # get logits for the last token in each row
        probs = torch.softmax(logits, dim=-1)
        token_id_next = torch.argmax(probs, dim=-1, keepdim=True)
        token_ids = torch.cat([token_ids, token_id_next], dim=-1)   # append predicted token to the input sequence
    return token_ids