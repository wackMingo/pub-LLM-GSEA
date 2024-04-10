from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import time
import json

def variables():
    """
    Script to generate results for multiple local models on multiple gene sets
    """
    # Define the models you want to test
    models1=[
        "BioGPT-Large",
        "BioGPT-Large-PubMedQA",
        "biogpt",
        "BioMedLM",
        "BioMedGPT-LM-7B"
    ] 
    models1 = dict(zip(models1, models1))
    
    models_custom_block={
        "PubMedQA_normal_model": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA",
        "PubMedQA_finetuned": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_new_vocab_save_after_train_custom_block_False",
        "PubMedQA_block_size_124": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_new_vocab_save_after_train_custom_block_True_124",
        "PubMedQA_block_size_253": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_new_vocab_save_after_train_custom_block_True_253",
        "PubMedQA_block_size_494": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_new_vocab_save_after_train_custom_block_True_494",
        "PubMedQA_block_size_754": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_new_vocab_save_after_train_custom_block_True_754",
        "PubMedQA_block_size_1045": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_new_vocab_save_after_train_custom_block_True_1045"
    }

    models_etc={
        "PubMedQA_normal": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA",
        "PubMedQA_expandvocab_false_newdata": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_save_after_train_custom_block_False_expandvocab_False", #with the new shorter line
        "PubMedQA_expandvocab_true_olddata": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_new_vocab_save_after_train_custom_block_False",
    }

    models_try_different_training_data = {
        "val1": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile1_date04/04/2024",
        "val2": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile2_date05/04/2024",
        "val3": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile3_date05/04/2024",
        "val4": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile4_date05/04/2024",
        "val5_2": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile5_n_genes2_date09/04/2024",
        "val5_3": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile5_n_genes3_date09/04/2024"
    }

    # The prompt you want to use for all models
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

    #nested list of gene sets to test
    nested_list = [
        ["UBE2I", "TRIP12", "MARCHF7"],
        ["MUC21", "MARCHF7", "HLA-DRB4"],
        ["TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3"],
        ["TTC21B", "HPS4", "LOC100653049", "CCDC39", "HECW2", "UBE2I"],
        ["TTC21B", "ZNF275", "UBE2I", "BRPF1", "OVOL3"],
        ["VARS2", "POLR2J3", "SEM1", "TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3", "MLYCD", "PRPF18"],
        ["TRIP12", "TTC21B", "MYLIP", "MARCHF7", "LOC100653049", "CELF1", "SRGAP2B", "CCDC39", "HECW2", "BRPF1", "OVOL3"]
    ]
    return models1, prompt_og, nested_list

# Function to load a model, run the prompt, and get back generated text
def test_model(model, tokenizer, prompt, time_list):
    start_time  = time.time()
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
    end_time = time.time()
    duration = end_time - start_time
    time_list.append(duration)
    print(f"Generating result took {duration} seconds" )

    #returns only the generated part and not the prompt that generated the text, otherwise way higher semantic sim score
    return(generated_text).split("proteins")[-1], time_list

def make_plots(result_dic, file_bar, title_bar):
    """
    Makes boxplot plot
    """
    cols= 4
    rows = 2
    plt.figure(figsize=(20, 5 * rows))  # Adjust figure size as needed
    colors = ["skyblue", "lightgreen", "tan", "pink"]
    colors = colors*int((int(len(list(result_dic.keys())))/4))
    values = list(result_dic.values())
    ticks = list(result_dic.keys())
    bplot1 = plt.boxplot(values, positions=range(len(values)), patch_artist=True)
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot1['medians']:
        median.set_color("red")
    plt.xlabel("Varying prompts and models")
    plt.ylabel("Semantic score")
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(range(len(ticks)), ticks, rotation=-65, ha='left', va='top', rotation_mode='anchor')
    plt.subplots_adjust(wspace=1.2, hspace=1)
    plt.title(title_bar)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_bar, format="png")

def plots_line(results_dic, file_line, title_line):
    """
    Makes line plot for local models vs GPT4 ground truth
    """
    data = results_dic
    plt.figure(figsize=(10, 6))

    # Plotting each model in the same figure
    for model, scores in data.items():
        plt.plot(scores, label=model)

    plt.xlabel('Gene set index')
    plt.ylabel('Similarity scores')
    #plt.title('Open source models versus GPT4 ground truth data set')
    plt.xticks(range(len(scores)), [f'Gene set {i+1}' for i in range(len(scores))])
    plt.legend()
    plt.title(title_line)
    plt.savefig(file_line, format="png")

def calculate_similarity(sentences, model):
    embedding_one = model.encode(sentences[0], convert_to_tensor=True)
    embedding_two = model.encode(sentences[1], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_one, embedding_two)
    return score

def sem_sim(big_list, model, models, file_n, file_write):
    #structuur dict: gene set als key, values is list met op volgorde alle scores van modellen
    result_dict = {}

    #open file to write away all results
    file_csv = open(file_write, "w")
    writer = csv.writer(file_csv, delimiter="|")
    writer.writerow(["model_name", "local", "gpt4", "score"])

    #load gpt4-Turbo ground truth data set
    with open(file_n) as f_in:
        gpt4_dict = json.load(f_in)

    for geneset in big_list:
        #x is hier lijst van alle antwoorden van alle modellen op 1 gene set
        #x is dus een dict key, dit is een gene set
        gpt4_answers = gpt4_dict[geneset]
        n_models = 0
        for y in big_list[geneset]:
            #y is hier alle antwoorden van 1 model op 1 gene set
            #y is dus een lijst, met alle antwoorden van 1 model, op 1 gene set
            score_for_model = 0
            n_ans = 0
            model_name = models[n_models]
            n_models += 1
            for z in y:
                #z is dus elk los antwoord van elk model van elke gene set van de hele lijst van alle interferences
                gpt4_sum_scores = 0
                counter_gpt4 = 0
                #this list has 15 answers from gpt4 which are all different(or we want them to be)
                #compare the result against all those from gpt4 and get an average score
                for gpt4_answer in gpt4_answers:
                    gpt4_comp = calculate_similarity([z, gpt4_answer], model).item()
                    writer.writerow([model_name, z, gpt4_answer, gpt4_comp])
                    gpt4_sum_scores += gpt4_comp
                    counter_gpt4 += 1
                score = gpt4_sum_scores / counter_gpt4
                score_for_model += score
                n_ans+=1
        
            #gemiddelde score van 1 model op 1 gene set van alle x keren interferen
            final_score = score_for_model / n_ans
            if result_dict.get(model_name) == None:
                result_dict[model_name] = [final_score]
            else:
                result_dict[model_name].append(final_score)
    return result_dict

def main():
    #file for ground truth
    file_n = "/exports/sascstudent/svanderwal2/programs/test_new_models/gpt_ground_truth.txt"
    #how many iterations should be done for each model and gene set, more iterations, more secure score
    iterations = 2
    #fiel to write away results of local models vs GPT4-Turbo ground truth with similarity score
    file_write = "testing_multi_result_gpt4_vs_local.csv"
    #file to write away line plot
    file_line = "line_all.png"
    title_line = "Performance of 5 local models on a GSEA task, 1 iterations"
    #file to write away bar plot
    file_bar = "bar_all.png"
    title_bar = "Performance of 5 local models on a GSEA task, 1 iterations"


    models, prompt_og, nested_list = variables()
    result_dic = {}
    for model in models.values():
        time_list = []
        loaded_model = AutoModelForCausalLM.from_pretrained(model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        for geneset in nested_list:
            geneset = " ".join(geneset)
            prompt = prompt_og % (geneset)
            result_list = []
            for x in range(iterations):
                result, time_list = test_model(loaded_model, tokenizer, prompt, time_list)
                result_list.append(result)
            if result_dic.get(geneset) == None:
                result_dic[geneset] = [result_list]
            else:
                result_dic[geneset].append(result_list)
        print("\n",model, ":", str(time_list))

    name_m = "all-mpnet-base-v2"
    model = SentenceTransformer(name_m)
    rs_dic = sem_sim(result_dic, model, list(models.keys()), file_n, file_write)
    make_plots(rs_dic, file_bar, title_bar)
    plots_line(rs_dic, file_line, title_line)

main()

