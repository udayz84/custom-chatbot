import tiktoken

def chunk_text(text, chunk_size=1500, overlap=50):
    tokenizer = tiktoken.get_encoding("cl100k_base") 
    tokens = tokenizer.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk))

    return chunks




def truncate_text(text, max_tokens = 4000):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])