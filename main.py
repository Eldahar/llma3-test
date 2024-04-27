from transformers import AutoModelForCausalLM, AutoTokenizer
#import transformers
import torch
import dotenv
from huggingface_hub import login

#login()

def main():
    model_id = "meta-llama/Meta-Llama-3-8B"

    #pipeline = transformers.pipeline(
    #    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
    #)

    #pipeline("Hey how are you doing today?")
    # Modell és tokenizer betöltése
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    #model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

    # Teszt szöveg
    text = "Hello, how can I help you today?"
    # Szöveg tokenizálása
    inputs = tokenizer.encode(text, return_tensors="pt")

    # Modell futtatása és válasz generálása
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
