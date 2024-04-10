from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import csv
import json

def run(system, user):
  client = OpenAI()
  completion = client.chat.completions.create(
    model="gpt-4-0125-preview",
    messages=[
      {"role": "system", "content": system},
      {"role": "user", "content": user}
    ]
  )

  return completion.choices[0].message.content

def vars():
  nested_list = [
        ["UBE2I", "TRIP12", "MARCHF7"],
        ["MUC21", "MARCHF7", "HLA-DRB4"],
        ["TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3"],
        ["TTC21B", "HPS4", "LOC100653049", "CCDC39", "HECW2", "UBE2I"],
        ["TTC21B", "ZNF275", "UBE2I", "BRPF1", "OVOL3"],
        ["VARS2", "POLR2J3", "SEM1", "TRIP12", "ZNF275", "TTC21B", "CELF1", "UBE2I", "BRPF1", "OVOL3", "MLYCD", "PRPF18"],
        ["TRIP12", "TTC21B", "MYLIP", "MARCHF7", "LOC100653049", "CELF1", "SRGAP2B", "CCDC39", "HECW2", "BRPF1", "OVOL3"]]
  system = "You are an efficient and insightful assistant to a molecular biologist"

  content_list={"normal": """
    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the most prominent biological process performed by the system

    Put the name at the top of the analysis as 'Process: <name>

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual; do not editorialize.
    For each important point, describe your reasoning and supporting information.

    Here are the interacting proteins: UBE2I TRIP12 MARCHF7
  """,
  "g;profiler":"""
      Write a critical analysis of the biological processes performed
      by this system of interacting proteins. Propose a brief name
      for the most prominent biological process performed by the system

      Put the name at the top of the analysis as 'Process: <name>

      Be concise, do not use unnecessary words. Be specific; avoid overly general
      statements such as 'the proteins are involved in various cellular processes'
      Be factual; do not editorialize.
      For each important point, describe your reasoning and supporting information.

      Here are the interacting proteins: UBE2I TRIP12 MARCHF7

      Given are some examples to which pathways these genes belong, from GO:BP
      Pathway 1: proteolysis involved in protein catabolic process
      Pathway 2: protein modification by small protein conjugation
      Pathway 3: protein catabolic process
      Pathway 4: protein modification by small protein conjugation or removal
""",
"top10":"""
      Write a critical analysis of the biological processes performed
      by this system of interacting proteins. Propose a brief name
      for the top 10 most prominent biological process performed by the system

      Be concise, do not use unnecessary words. Be specific; avoid overly general
      statements such as 'the proteins are involved in various cellular processes'
      Be factual; do not editorialize.
      For each important point, describe your reasoning and supporting information.

      Here are the interacting proteins: UBE2I TRIP12 MARCHF7
"""}


  return system, content_list, nested_list

def calculate_similarity(sentences, model):
    embedding_one = model.encode(sentences[0], convert_to_tensor=True)
    embedding_two = model.encode(sentences[1], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_one, embedding_two)
    return score

def gen_results(result_dic):
  name_m = "all-mpnet-base-v2"
  model = SentenceTransformer(name_m)
  for key in result_dic:
    already_calculated_list = []
    prompt_scores = []
    result_line = result_dic[key]
    frozen_it = 0
    for result_frozen in result_line:
      frozen_it+=1
      variable_it = 0
      for result_variable in result_line:
        variable_it+=1
        if result_frozen != result_variable and str(frozen_it)+":"+str(variable_it) not in already_calculated_list:
          semantic_score = calculate_similarity([result_frozen, result_variable], model)
          prompt_scores.append(semantic_score.item())
          already_calculated_list.append(str(frozen_it)+":"+str(variable_it))
          already_calculated_list.append(str(variable_it)+":"+str(frozen_it))
    #result_dic[key] = sum(prompt_scores)/len(prompt_scores)
    result_dic[key] = prompt_scores
  return result_dic

def make_plot_chatgpt_comparison(dic):
    cols= 4
    rows = 2  # Calculate rows needed
    plt.figure(figsize=(20, 5 * rows))  # Adjust figure size as needed

    values = list(dic.values())
    ticks = list(dic.keys())
    colors = ["blue", "black", "green", "yellow"]
    colors = colors*2
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
    #plt.tight_layout()
    plt.title("Consistency of OpenAI GPT models with 4 different prompts and 2 models")
    #plt.title("Consistency of ChatGPT models on different prompts")
    plt.grid(True)
    plt.savefig("chatgpt_compare.png", format="png")

def comp(system, content_list):
  result_dic={}
  for key in content_list:
    result_dic[key] = []

  for iteration in range(15):
    print(f"Iteration {iteration},", end=" ")
    for key in content_list:
      prompt = content_list[key]
      print(key) 
      result = run(system, prompt)
      result_dic[key].append(result)

  big_list = []
  for key in result_dic:
     write_p = [key]
     write_p = write_p + result_dic[key]
     print(write_p)
     big_list.append(write_p)
  print(big_list)

  with open("gpt4_comparison_result_dictionary.csv", "w", newline='') as file:
      writer = csv.writer(file, delimiter="|")
      writer.writerows(big_list)

  scores_dic = gen_results(result_dic)
  print(scores_dic)
  make_plot_chatgpt_comparison(scores_dic)

def make_truthset(genes_list, system):
  prompt = """
    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the top 10 most prominent biological process performed by the system

    Put the name at the top of the analysis as 'Process: <name>

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual; do not editorialize.
    For each important point, describe your reasoning and supporting information.

    Here are the interacting proteins: %s
  """
   #gpt_message = run(prompt, system)

  result_dic = {}
  for gene_set in genes_list:
      result_dic[" ".join(gene_set)] = []
      for x in range(15):
        prompt_new = prompt % (" ".join(gene_set))
        gpt_message = run(prompt_new, system)
        result_dic[" ".join(gene_set)].append(gpt_message)
  
  with open("gpt_ground_truth.txt", "w") as conv_file:
    conv_file.write(json.dumps(result_dic))

   

def main():
  system, content_list, genes_list = vars()
  compare = False

  if compare == True:
    comp(system, content_list)

  else:
    make_truthset(genes_list, system)



main()