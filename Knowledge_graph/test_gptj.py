from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", pad_token_id= tokenizer.eos_token_id)

def generate_object(entity_label, relation_label):
    # Define the prompt with placeholders for the entity and relation labels
    prompt = f"""Based on the information provided, please answer the following question:
    Q: Monte Cremasco # country
    A: Italy
    Q: Johnny Depp # children
    A: Jack Depp, Lily-Rose Depp
    Q: Wolfgang Sauseng # employer
    A: University of Music and Performing Arts Vienna
    Q: {entity_label} # {relation_label}
    A:"""
    
    # Encode the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Calculate the appropriate max_length by adding some buffer to the input length
    max_length = input_ids.shape[1] + 50  # Adding 50 tokens as buffer
    gen_tokens = model.generate(
        input_ids,
        max_length=max_length,
    )
    full_response = tokenizer.batch_decode(gen_tokens)[0]
    
    # Extract the relevant answer
    query_string = f"Q: {entity_label} # {relation_label}\n    A:"
    start_index = full_response.find(query_string)
    if start_index != -1:
        start_index += len(query_string)
        end_index = full_response.find("\n    Q:", start_index)
        if end_index == -1:  # No more questions, take till end
            end_index = len(full_response)
        answer = full_response[start_index:end_index].strip()
    else:
        answer = "No answer found."

    # Print the answer for debugging
    print("Generated response:", answer)
    
    return answer


# Example usage of the function
if __name__ == "__main__":
    entity = "Barack Obama"
    relation = "spouse"
    generate_object(entity, relation)

