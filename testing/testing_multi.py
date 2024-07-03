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
import scipy.stats as stats


def variables(test):
    """
    Script to generate results for multiple local models on multiple gene sets
    """
    # Define the models you want to test
    models1={
        "BioGPT-large": "/exports/sascstudent/svanderwal2/programs/test_new_models/BioGPT-Large",
        "PubMedQA": "/exports/sascstudent/svanderwal2/programs/test_new_models/BioGPT-Large-PubMedQA",
        "biogpt": "/exports/sascstudent/svanderwal2/programs/test_new_models/biogpt",
        "BioMedLM": "/exports/sascstudent/svanderwal2/programs/test_new_models/BioMedLM",
        "BioMedGPT": "/exports/sascstudent/svanderwal2/programs/test_new_models/BioMedGPT-LM-7B"
    }

    models_custom_block={
        "PubMedQA_normal": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA",
        "PubMedQA_finetuned": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_05_06_2024",
        "PubMedQA_block_size_124": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_124_date_27_05_2024_onegene_morepaths",
        "PubMedQA_block_size_253": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_253_date_27_05_2024_onegene_morepaths",
        "PubMedQA_block_size_494": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_494_date_28_05_2024_onegene_morepaths",
        "PubMedQA_block_size_754": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_754_date_28_05_2024_onegene_morepaths",
        "PubMedQA_block_size_1020": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_1020_date_01_06_2024_onegene_morepaths"
    }

    models_custom_block_ind1 = {
        "PubMedQA_normal": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA",
        "PubMedQA_finetuned" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_02_06_2024",
        "PubMedQA_block_size_124" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_124_date_03_06_2024_path_gene_desc",
        "PubMedQA_block_size_253":  "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_253_date_03_06_2024_path_gene_desc",
        "PubMedQA_block_size_494" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_494_date_03_06_2024_path_gene_desc",
        "PubMedQA_block_size_754" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_754_date_04_06_2024_path_gene_desc",
        "PubMedQA_block_size_1020": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_True_1020_date_04_06_2024_path_gene_desc"
    }

    models_etc={
        "PubMedQA_normal": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA",
        "PubMedQA_expandvocab_false_newdata": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_save_after_train_custom_block_False_expandvocab_False", #with the new shorter line
        "PubMedQA_expandvocab_true_olddata": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_one_path_one_gene_new_vocab_save_after_train_custom_block_False",
    }

    models_try_different_training_data = {
        "PubMedQA_normal": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA",
        "PathGeneDesc": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile1_date04/04/2024",
        "PathGenes": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile2_date05/04/2024",
        "PathGene": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile3_date05/04/2024",
        "PathGeneDescSyno": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile4_date05/04/2024",
        "Path2Genes": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile5_n_genes2_date09/04/2024",
        "Path3Genes": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_expandvocab_False_trainingfile5_n_genes3_date09/04/2024",
        "GenePaths": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_onegene_morepaths_date_26_05_2024"
    }

    same_model = {
        "v1" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_09_06_2024",
        "v2" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_10_06_2024_v2",
        "v3" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_12_06_2024_v3",
        "v4" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_14_06_2024_v4",    
        "v5" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_14_06_2024_v5",
        "v6" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_14_06_2024_v6",
        "v7" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_14_06_2024_v7",
        "v8" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_16_06_2024_v8",
        "v9" : "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large/BioGPT-Large-PubMedQA_finetuned_1_epochs_custom_block_False_path_gene_desc_date_16_06_2024_v9",
        "v10": "/exports/sascstudent/svanderwal2/programs/test_new_models/training_large"}


    # The prompt you want to use for all models
    prompt_og="""
    You are an efficient and insightful assistant to a molecular biologist

    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the top 10 most prominent biological processes performed by the system

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
    
    if test == "8structures":
        return_models_dic = models_try_different_training_data
    elif test == "5base":
        return_models_dic = models1
    elif test == "custom_onegene_morepaths":
        return_models_dic = models_custom_block
    elif test == "custom_pathgenedesc":
        return_models_dic = models_custom_block_ind1

    return return_models_dic, prompt_og, nested_list

def calculate_split_size(tensor, max_split_size_mb):
    bytes_per_element = tensor.element_size()
    max_split_size_bytes = max_split_size_mb * 1024 * 1024
    elements_per_split = max_split_size_bytes // bytes_per_element
    return elements_per_split

# Function to load a model, run the prompt, and get back generated text
def test_model(model, tokenizer, prompt, time_list, max_split_size_mb=128):
    torch.cuda.empty_cache()

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

    #split inputs into smaller chunks, video card does not have enough memory for very large models and loading inputs
    split_size = calculate_split_size(inputs['input_ids'], max_split_size_mb)
    input_splits = torch.split(inputs['input_ids'], split_size, dim=1)

    outputs = []
    for split in input_splits:
        split_inputs = {k: v[:, :split.size(1)] for k, v in inputs.items()}
        output_sequences = model.generate(**split_inputs, max_length=500, num_return_sequences=20, do_sample=True)
        for output_sequence in output_sequences:
            generated_text = tokenizer.decode(output_sequence, skip_special_tokens=True).split("proteins")[-1]
            outputs.append(generated_text)

    #output_sequences = model.generate(**inputs, max_length=500, num_return_sequences=20, do_sample=True)

    outputs = []
    #decoding output sequences
    for output_sequence in output_sequences:
        generated_text = tokenizer.decode(output_sequence, skip_special_tokens=True).split("proteins")[-1]
        outputs.append(generated_text)

    end_time = time.time()
    duration = end_time - start_time
    time_list.append(duration)
    print(f"Generating result took {duration} seconds" )

    #returns only the generated part and not the prompt that generated the text, otherwise way higher semantic sim score
    return outputs, time_list

def t_test(result_dic):
    normal_prompt_result = result_dic[list(result_dic.keys())[0]]
    pvals = {}
    for x in result_dic:
        pval = stats.ttest_ind(normal_prompt_result, result_dic[x], equal_var = True).pvalue
        pvals[x] = pval
    return pvals
    


def make_plots(result_dic, file_bar, title_bar, colors):
    """
    Makes boxplot plot
    """

    cols= 4
    rows = 2
    title_font=18
    yticks_font = 14
    xticks_font = 20
    xlabel_font = 18
    ylabel_font = 16
    pval_font = 12

    fig = plt.figure(figsize=(15, 5 * rows))  # Adjust figure size as needed
    ax = fig.add_subplot()

    values = list(result_dic.values())
    ticks = list(result_dic.keys())
    bplot1 = ax.boxplot(values, positions=range(len(values)), patch_artist=True)

    """
    #adding pvals
    pval_dict = t_test(result_dic)
    for ind, x in enumerate(values):
        print(len(x))
        ax.text(ind, np.median(x)-0.001, "pval: "+str("{:.3e}".format((list(pval_dict.values())[ind]))), ha="center", va="top", fontsize=pval_font)
    """

    for ind, patch in enumerate(bplot1['boxes']):
        patch.set_facecolor(colors[ind])
    for median in bplot1['medians']:
        median.set_color("black")



    ax.set_xlabel("Model training data identifier", fontsize=xlabel_font)
    ax.set_ylabel("Semantic score", fontsize=ylabel_font)
    #plt.subplots_adjust(bottom=0.2)
    ax.xaxis.set_ticks(range(len(ticks)), ticks, rotation=-65, ha='left', va='top', rotation_mode='anchor', fontsize=xticks_font)
    ax.yaxis.set_tick_params(labelsize=yticks_font)
    plt.subplots_adjust(wspace=1.2, hspace=1)
    plt.title(title_bar, fontsize=title_font)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_bar, format="png")

def plots_line(results_dic, file_line, title_line, colors):
    """
    Makes line plot for local models vs GPT4 ground truth
    """
    data = results_dic
    plt.figure(figsize=(10, 6))

    count = 0
    # Plotting each model in the same figure
    for model, scores in data.items():
        new_scores = []
        #print(len(scores))
        for pos in range(0, len(scores), 300):
            new_scores.append(sum(scores[pos:pos+300]) / 300)
        #print(new_scores)
        plt.plot(new_scores, label=model, color=colors[count])
        count+=1

    plt.xlabel('Gene set index')
    plt.ylabel('Similarity scores')
    #plt.title('Open source models versus GPT4 ground truth data set')
    plt.xticks(range(7), [f'Gene set {i+1}' for i in range(7)], fontsize=10)
    plt.legend()
    plt.title(title_line)
    plt.savefig(file_line, format="png")

def calculate_similarity(sentences, model):
    embedding_one = model.encode(sentences[0], convert_to_tensor=True)
    embedding_two = model.encode(sentences[1], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_one, embedding_two)
    return score

def sem_sim(big_list, model, models, file_n, file_write, file_write2, f_out2):
    #structuur dict: gene set als key, values is list met op volgorde alle scores van modellen
    result_dict = {}

    #open file to write away all results
    file_csv = open(file_write, "w")
    file_csv2 = open(file_write2, "w")
    writer = csv.writer(file_csv, delimiter="|")
    writer2 = csv.writer(file_csv2, delimiter="|")
    writer.writerow(["model_name", "local", "gpt4", "score"])
    writer2.writerow(["geneset", "model_name", "response"])

    #load gpt4-Turbo ground truth data set
    with open(file_n) as f_in:
        gpt4_dict = json.load(f_in)

    for geneset in big_list:
        #x is hier lijst van alle antwoorden van alle modellen op 1 gene set
        #x is dus een dict key, dit is een gene set
        gpt4_answers = gpt4_dict[geneset]
        n_models = 0
        for y in big_list[geneset]:
            list_per_model = []
            print(y)
            for x in y:
                #y is hier alle antwoorden van 1 model op 1 gene set
                #y is dus een lijst, met alle antwoorden van 1 model, op 1 gene set
                model_name = models[n_models]
                n_models += 1
                for z in x:
                    #z is dus elk los antwoord van elk model van elke gene set van de hele lijst van alle interferences
                    #this list has 15 answers from gpt4 which are all different(or we want them to be)
                    #compare the result against all those from gpt4 and get an average score
                    writer2.writerow([geneset, model_name, z])
                    all_gpt_scores_for_sum = []
                    for gpt4_answer in gpt4_answers:
                        gpt4_comp = calculate_similarity([z, gpt4_answer], model).item()
                        writer.writerow([model_name, z, gpt4_answer, gpt4_comp])
                        all_gpt_scores_for_sum.append(gpt4_comp)
                    list_per_model.extend(all_gpt_scores_for_sum)
            
                if result_dict.get(model_name) == None:
                    result_dict[model_name] = list_per_model
                else:
                    result_dict[model_name].extend(list_per_model)

    with open(f_out2, "w") as file:
        file.write(json.dumps(result_dict))
    return result_dict

def main():
    #file for ground truth
    file_n = "/exports/sascstudent/svanderwal2/programs/test_new_models/gpt_ground_truth.txt"

    tests = []
    tests.append("5base")
    tests.append("8structures")
    tests.append("custom_onegene_morepaths")
    tests.append("custom_pathgenedesc")
    part = 1

    for test in tests:
        if test == "8structures":
            models, prompt_og, nested_list = variables(test)

            #files for GPT4 vs local models, 8 different data structures
            file_write = "testing_multi_result_gpt4_vs_local_8structures.csv"
            file_write2 = "testing_multi_result_local_8structures.csv"
            f_out2 = "testing_multi_semantic_sim_dictionary_scores_8strucutres.csv"
            #file to write away line plot
            file_line = "line_all_8structures.png"
            title_line = "Performance of 8 different fine-tuned models on a GSEA task"
            #file to write away bar plot
            file_bar = "bar_all_8structures.png"
            title_bar = "Performance of 8 different fine-tuned models on a GSEA task"

        elif test == "5base":
            models, prompt_og, nested_list = variables(test)

            #files for the 5 base models
            file_write = "testing_multi_result_gpt4_vs_local_5base.csv"
            file_write2 = "testing_multi_result_local_5base.csv"
            f_out2 = "testing_multi_semantic_sim_dictionary_scores_5base.csv"
            #file to write away line plot
            file_line = "line_all_5base.png"
            title_line = "Performance of 5 base models on a GSEA task"
            #file to write away bar plot
            file_bar = "bar_all_5base.png"
            title_bar = "Performance of 5 base models on a GSEA task"

        elif test == "custom_onegene_morepaths":
            models, prompt_og, nested_list = variables(test)

            #files for the 5 base models
            file_write = "testing_multi_result_gpt4_vs_local_custom_onegene_morepaths.csv"
            file_write2 = "testing_multi_result_local_custom_onegene_morepaths.csv"
            f_out2 = "testing_multi_semantic_sim_dictionary_scores_custom_onegene_morepaths.csv"
            #file to write away line plot
            file_line = "line_all_custom_onegene_morepaths.png"
            title_line = "Performance of 8 fine-tuned models on a GSEA task - GenePaths"
            #file to write away bar plot
            file_bar = "bar_all_custom_onegene_morepaths.png"
            title_bar = "Performance of 8 fine-tuned models on a GSEA task - GenePaths"

        elif test == "custom_pathgenedesc":
            models, prompt_og, nested_list = variables(test)

            #files for the 5 base models
            file_write = "testing_multi_result_gpt4_vs_local_custom_pathgenedesc.csv"
            file_write2 = "testing_multi_result_local_custom_pathgenedesc.csv"
            f_out2 = "testing_multi_semantic_sim_dictionary_scores_custom_pathgenedesc.csv"
            #file to write away line plot
            file_line = "line_all_custom_pathgenedesc.png"
            title_line = "Performance of 8 fine-tuned models on a GSEA task - PathGeneDesc"
            #file to write away bar plot
            file_bar = "bar_all_custom_onegene_pathgenedesc.png"
            title_bar = "Performance of 8 fine-tuned models on a GSEA task - PathGeneDesc"
    


        if part == 1:
            result_dic = {}
            for model in models.values():
                torch.cuda.empty_cache()
                time_list = []
                if model == "/exports/sascstudent/svanderwal2/programs/test_new_models/BioMedGPT-LM-7B":
                    loaded_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.float16)
                else:
                    loaded_model = AutoModelForCausalLM.from_pretrained(model)               
                tokenizer = AutoTokenizer.from_pretrained(model)
                for geneset in nested_list:
                    geneset = " ".join(geneset)
                    prompt = prompt_og % (geneset)
                    result_list = []
                    outputs, time_list = test_model(loaded_model, tokenizer, prompt, time_list)
                    result_list.append(outputs)
                    if result_dic.get(geneset) == None:
                        result_dic[geneset] = [result_list]
                    else:
                        result_dic[geneset].append(result_list)
                print("\n",model, ":", str(time_list))

            name_m = "BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
            #name_m = "all-mpnet-base-v2"
            model = SentenceTransformer(name_m)
            print(result_dic)
            rs_dic = sem_sim(result_dic, model, list(models.keys()), file_n, file_write, file_write2, f_out2)
            part = 2
        if part == 2:
            #https://colorswall.com/palette/171311
            colors = ["#9b5fe0", "#16a4d8", "#60dbe8", "#8bd346", "#efdf48", "#f9a52c", "#d64e12", "#ff00ff"]
            with open(f_out2, "r") as file:
                rs_dic = json.load(file)
            #first read file here for the scores
            make_plots(rs_dic, file_bar, title_bar, colors)
            plots_line(rs_dic, file_line, title_line, colors)

main()

