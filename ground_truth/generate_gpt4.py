from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import csv
import json
import pandas as pd
import scipy.stats as stats

def run(user):
  client = OpenAI()
  completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
      {"role": "user", "content": user}
    ],
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
  
  content_list_removed = {"baseline": """
    You are an efficient and insightful assistant to a molecular biologist

    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the most prominent biological process performed by the system
    Put the name at the top of the analysis as 'Process: <name>

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual; do not editorialize.
    For each important point, describe your reasoning and supporting information.

    Here are the interacting proteins: %s
  """,
  "no_be_factual": """
    You are an efficient and insightful assistant to a molecular biologist
    
    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the most prominent biological process performed by the system

    Put the name at the top of the analysis as 'Process: <name>

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    For each important point, describe your reasoning and supporting information.

    Here are the interacting proteins: %s
    """,
  "no_for_each": """
    You are an efficient and insightful assistant to a molecular biologist

    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the most prominent biological process performed by the system

    Put the name at the top of the analysis as 'Process: <name>

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual; do not editorialize.

    Here are the interacting proteins: %s
    """,
  "no_be_concise": """
    You are an efficient and insightful assistant to a molecular biologist

    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the most prominent biological process performed by the system

    Put the name at the top of the analysis as 'Process: <name>

    Be factual; do not editorialize.
    For each important point, describe your reasoning and supporting information.

    Here are the interacting proteins: %s
  """,
  "top10":"""
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
""",
  "top10_relaxed":"""
    You are an efficient and insightful assistant to a molecular biologist

    Propose a brief name for the top 10 most prominent biological processes performed by the system

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual; do not editorialize.

    Here are the interacting proteins: %s
""",
  "no_critical": """
    You are an efficient and insightful assistant to a molecular biologist

    Propose a brief name for the most prominent 
    biological process performed by the system
    Put the name at the top of the analysis as 'Process: <name>

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual; do not editorialize.
    For each important point, describe your reasoning and supporting information.

    Here are the interacting proteins: %s
  """,
  "no_persona": """
    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name
    for the most prominent biological process performed by the system

    Put the name at the top of the analysis as 'Process: <name>

    Be concise, do not use unnecessary words. Be specific; avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual; do not editorialize.
    For each important point, describe your reasoning and supporting information.

    Here are the interacting proteins: %s
  """}

  return content_list_removed, nested_list

def calculate_similarity(sentences, model):
    embedding_one = model.encode(sentences[0], convert_to_tensor=True)
    embedding_two = model.encode(sentences[1], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_one, embedding_two)
    return score

def gen_results(result_dic):
  print("--Comparing--")
  name_m = "all-mpnet-base-v2"
  model = SentenceTransformer(name_m)
  for key in result_dic:
    print(key)
    scores_per_geneset_list = []
    for geneset_key in result_dic[key]:
      print(geneset_key, len(result_dic[key][geneset_key]))
      already_calculated_list = []
      response_scores = []
      result_line = result_dic[key][geneset_key]
      frozen_variable = 0
      for result_frozen in result_line:
        #result_frozen_nested_list is every answer on that specific gene set for that specific prompt
        variable_variable=0
        for result_variable in result_line:
          if variable_variable != frozen_variable and str(frozen_variable)+":"+str(variable_variable) not in already_calculated_list:
            semantic_score = calculate_similarity([result_frozen, result_variable], model)
            print(f"------Frozen | var: {frozen_variable}\n{[result_frozen]}")
            print(f"------Variable| var: {variable_variable}\n{[result_variable]}")
            print("Score: "+str(semantic_score.item()))
            scores_per_geneset_list.append(semantic_score.item())
            already_calculated_list.append(str(frozen_variable)+":"+str(variable_variable))
            already_calculated_list.append(str(variable_variable)+":"+str(frozen_variable))
          variable_variable+=1
        frozen_variable+=1
      #add average to the list
      #scores_per_geneset_list.append(sum(response_scores)/len(response_scores)) #this instead of appending to scores_per_geneset
    print(scores_per_geneset_list)
    result_dic[key] = scores_per_geneset_list
  print(result_dic)

  for x in result_dic:
    print(result_dic[x])

  return result_dic


def make_plot_chatgpt_comparison(dic, pval_dict):
  colors = ["#9b5fe0", "#16a4d8", "#60dbe8", "#8bd346", "#efdf48", "#f9a52c", "#d64e12", "#ff00ff"]
  #print(dic)
  cols= 4
  rows = 2  # Calculate rows needed
  fig = plt.figure(figsize=(20, 5 * rows))  # Adjust figure size as needed
  ax = fig.add_subplot()

  values = list(dic.values())
  ticks = list(dic.keys())

  title_font=18
  yticks_font = 14
  xticks_font = 22
  xlabel_font = 18
  ylabel_font = 16
  pval_font = 14

  """
  for ind, x in enumerate(values):
    pval = list(pval_dict.values())[ind]
    ax.text(ind, np.median(x)-0.001, "pval: "+str("{:.3e}".format(pval)), ha="center", va="top", fontsize=pval_font)
  """
  color = "#1b9e77"
  bplot1 = ax.boxplot(values, positions=range(len(values)), patch_artist=True, labels=values)

  for ind, patch in enumerate(bplot1['boxes']):
      patch.set_facecolor(colors[ind])
  for median in bplot1['medians']:
      median.set_color("black")

  ax.set_xlabel("Special prompt identification", fontsize=xlabel_font)
  ax.set_ylabel("Semantic score", fontsize=ylabel_font)
  plt.subplots_adjust(bottom=0.2)
  ax.xaxis.set_ticks(range(len(ticks)), ticks, rotation=-65, ha='left', va='top', rotation_mode='anchor', fontsize=xticks_font)
  ax.yaxis.set_tick_params(labelsize=yticks_font)
  plt.subplots_adjust(wspace=1.2, hspace=1)
  plt.tight_layout()
  plt.title("Consistency of prompts on GPT4-Turbo API", fontsize=title_font)
  plt.grid(True)
  plt.savefig("chatgpt_compare.png", format="png", bbox_inches="tight")

def comp(content_list, genes_list):
  #if you want to limit amount of prompts do it here
  newdic = {}
  for key in list(content_list.keys()):
    newdic[key] = content_list[key]
  content_list = newdic

  genes_list = genes_list #remove this line for final
  #print(genes_list)
  result_dic={}
  for key in content_list:
    result_dic[key] = {}
    for geneset in genes_list:
      geneset = ' '.join(geneset)
      result_dic[key][geneset] = [] #fix this

  for iteration in range(20): #make into 20, before that fill up saldo
    print(f"Iteration {iteration+1}")
    for key in content_list:
      for geneset in genes_list:
        geneset = ' '.join(geneset)
        prompt_og = content_list[key]
        prompt = prompt_og % geneset
        #print(key, geneset) 
        result = run(prompt)
        #result=prompt
        result_dic[key][geneset].append(result)

  #dictionary with prompt=key then value=dic with key=geneset and value=list with responses

  #fix this with the new dictionary stuff
  big_list = []
  for key in result_dic:
     for geneset in result_dic[key]:
      write_p = [key] 
      write_l = [geneset]
      write_p = write_p + write_l + result_dic[key][geneset]
      #print(write_p)
      big_list.append(write_p)
  #print(big_list)

  with open("gpt4_comparison_result.csv", "w", newline='') as file:
      writer = csv.writer(file, delimiter="|")
      writer.writerows(big_list)

  scores_dic = gen_results(result_dic)

  with open("gpt4_comparison_scores_dictionary.csv", "w") as file:
    file.write(json.dumps(scores_dic))

def make_truthset(genes_list, file_save):
  #Write a critical analysis of the biological processes performed
  # by this system of interacting proteins. 
  prompt = """
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
  """
   #gpt_message = run(prompt, system)

  result_dic = {}
  print(genes_list)
  for gene_set in genes_list:
      result_dic[" ".join(gene_set)] = []
      for x in range(20):
        prompt_new = prompt % (" ".join(gene_set))
        gpt_message = run(prompt_new)
        print(gpt_message)
        result_dic[" ".join(gene_set)].append(gpt_message)
  
  with open(file_save, "w") as conv_file:
    conv_file.write(json.dumps(result_dic))

def t_test(gpt4_dict):
  pval_dict = {}
  normal_list = gpt4_dict[list(gpt4_dict.keys())[0]]
  for key in gpt4_dict:
    pval = stats.ttest_ind(normal_list, gpt4_dict[key], equal_var = True).pvalue
    print(pval)
    pval_dict[key] = pval
  return pval_dict
    
def main():
  content_list, genes_list = vars()
  compare = 3

  if compare == 1:
    #first select which prompts we want to test
    comp(content_list, genes_list)

  elif compare == 2:
    file_save = "gpt_ground_truth.txt"
    make_truthset(genes_list, file_save)

  if compare == 3:
    with open("gpt4_comparison_scores_dictionary.csv", "r") as file:
      gpt4_dict = json.load(file)
    pval_dict = t_test(gpt4_dict)
    #read file that is written away
    make_plot_chatgpt_comparison(gpt4_dict, pval_dict)




main()