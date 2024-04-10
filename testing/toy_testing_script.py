from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def test_model(model, tokenizer, prompt):
    #if cuda is avaiable device value is 0
    if torch.cuda.is_available():
        print("Cuda is available, moving model to CUDA to train on GPU")
        model.to("cuda")

    #input encoding, return as PyTorch Tensor
    inputs = tokenizer(prompt, return_tensors="pt")

    #if GPU avaiable move inputs to GPU
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    #generate text
    output_sequences = model.generate(**inputs, max_length=500, num_return_sequences=1)

    #decoding output sequences
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    #returns only the generated part and not the prompt that generated the text, otherwise way higher semantic sim score
    return(generated_text).split("proteins")[-1]

def main():
    prompt_og="""
        You are an efficient and insightful assistant to a molecular biologist

        Write a critical analysis of the biological processes performed
        by this system of interacting proteins. Propose a brief name
        for the top 10 most prominent biological process performed by the system

        Put the name at the top of the analysis as 'Process: <name>

        Be concise, do not use unnecessary words. Be specific; avoid overly general
        statements such as 'the proteins are involved in various cellular processes'
        Be factual; do not editorialize.
        For each important point, describe your reasoning and supporting information.

        Here are the interacting proteins: %s

        Process:
        """

    nested_list = [
        ["UBE2I", "TRIP12", "MARCHF7"],
        ["MUC21", "MARCHF7", "HLA-DRB4"],
        ["TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3"],
        ["TTC21B", "HPS4", "LOC100653049", "CCDC39", "HECW2", "UBE2I"],
        ["TTC21B", "ZNF275", "UBE2I", "BRPF1", "OVOL3"],
        ["VARS2", "POLR2J3", "SEM1", "TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3", "MLYCD", "PRPF18"],
        ["TRIP12", "TTC21B", "MYLIP", "MARCHF7", "LOC100653049", "CELF1", "SRGAP2B", "CCDC39", "HECW2", "BRPF1", "OVOL3"]
    ]

    prompt = prompt_og % " ".join(nested_list[0])
    model = "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA"
    loaded_model = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(test_model(loaded_model, tokenizer, prompt))

main()