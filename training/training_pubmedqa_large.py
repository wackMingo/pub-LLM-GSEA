from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import SFTTrainer
from peft import get_peft_model, PeftModelForCausalLM
import datetime
"""
Script to fine-tune models
"""


def example():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    text = """
Pathway Glutathione metabolism contains the gene GPX3 with gene description glutathione peroxidase 3 [Source:HGNC Symbol;Acc:HGNC:4555] and gene synonym@
Pathway Glutathione metabolism contains the gene GSTT2 with gene description glutathione S-transferase theta 2 (gene/pseudogene) [Source:HGNC Symbol;Acc:HGNC:4642] and gene synonym @
Pathway Glutathione metabolism contains the gene ANPEP with gene description alanyl aminopeptidase, membrane [Source:HGNC Symbol;Acc:HGNC:500] and gene synonym PEPN@
Pathway Glutathione metabolism contains the gene OPLAH with gene description 5-oxoprolinase, ATP-hydrolysing [Source:HGNC Symbol;Acc:HGNC:8149] and gene synonym OPLA@
Pathway Glutathione metabolism contains the gene GGT5 with gene description gamma-glutamyltransferase 5 [Source:HGNC Symbol;Acc:HGNC:4260] and gene synonym GGTLA1
"""
    text = text.split("@")

    tokenized = tokenizer(text, padding=True, truncation=True, return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]
    for num, x in enumerate(zip(text, offsets)):
        org_seq = x[0]
        splits = x[1]
        print(len(org_seq.split(" ")), org_seq.strip(("\n")))
        p = []
        for spl in splits:
            p.append(text[num][spl[0]:spl[1]])
        print(len(p), p)
        print(len(tokenized["input_ids"][num]), tokenized["input_ids"][num])
        print(len(tokenized["attention_mask"][num]), tokenized["attention_mask"][num])
        print()

def expand_vocab(model, tokenizer):
    make_custom_vocab_file = "make_custom_vocab.txt"
    list_vocab = []
    with open(make_custom_vocab_file, "r") as file:
        for x in file:
            list_vocab.append(x.strip("\n"))
    tokenizer.add_tokens(list_vocab)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def preprocess_data(model, file_name, tokenizer, custom_block, expand_vocab):
    if expand_vocab:
        model, tokenizer = expand_vocab(model, tokenizer)
    #max_length = tokenizer.model_max_length
    max_length_list = [124, 253, 494, 754, 1020]
    max_length = max_length_list[4]
    #only need to do [4] and i think [0] again with more time

    if custom_block == False:
        big_text = ""
        with open(file_name, "r") as text:
            for line in text:
                big_text+=(line.strip("\n").strip().strip(",")+"|")
        datasets = []
        for x in range(0, len(big_text), 1024):
            #print(big_text[x:x+1024].strip("\n"))
            datasets.append(big_text[x:x+1024].strip("\n"))
        data_dic = {}
        data_dic["input_ids"] = []
        data_dic["attention_mask"] = []
        data_dic["labels"] = []
        for x in datasets:
            #print(x)
            tokenized_data = tokenizer(x, padding="max_length", truncation=True)
            data_dic["input_ids"].append(tokenized_data["input_ids"])
            data_dic["attention_mask"].append(tokenized_data["attention_mask"])
            data_dic["labels"].append(tokenized_data["input_ids"])
        #print(data_dic)
        #dict in vorm van met 3 keys, elke value van key is een nested list met elke lijst een embedding van 1 zin gevuld tot max length zegmaar
        ds = Dataset.from_dict(data_dic)

    else:
        full_text = ""
        with open(file_name, "r") as text:
            for x in text:
                full_text+=x.strip("\n")+"</SEP>"
        #max length is the block size, try random block sizes
        blocked_text = {}
        blocked_text["input_ids"] = []
        blocked_text["attention_mask"] = []
        blocked_text["labels"] = []
        val=0
        for i in range(0,len(full_text),max_length):
            block = full_text[i:i+max_length]
            val +=1
            print(val, block)
            tokenized_block = tokenizer(block, padding="max_length", truncation=True)
            blocked_text["input_ids"].append(tokenized_block["input_ids"])
            blocked_text["attention_mask"].append(tokenized_block["attention_mask"])
            blocked_text["labels"].append(tokenized_block["input_ids"])
        print("Block length:", max_length)
        ds = Dataset.from_dict(blocked_text)

    return ds, model, tokenizer, max_length

def training_and_eval(datasets, model, m_name, tokenizer, file_name, lora, custom_block, max_length, expand_vocab, val):
    date = datetime.datetime.now().strftime("%d_%m_%Y")

    if lora == True:
        epochs=1
        special=f"one_path_one_gene_new_vocab_lora_{lora}_custom_block_{custom_block}_{max_length}"
        output_dir_name = f"./{m_name}_finetuned_{epochs}_epochs_{special}"
        #LoRa
        attention_modules = ["q_proj", "v_proj"]

        #If targeting all linear layers
        linear_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']

        max_token_id = len(tokenizer.get_vocab())
        model_vocab_size = model.config.vocab_size
        print(max_token_id, model_vocab_size)

        lora_config = LoraConfig(
        r=16,
        target_modules = attention_modules,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM")

        #LoRa
        training_args = TrainingArguments(
                output_dir= output_dir_name,
                num_train_epochs=epochs,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                #fp16=True,
                gradient_accumulation_steps=4,
                weight_decay=0.01,
                logging_dir="./logs",
                optim="adafactor"
        )
        #LoRa
        model = PeftModelForCausalLM(model, lora_config)
        max_token_id = len(tokenizer.get_vocab())
        model_vocab_size = model.config.vocab_size
        print("After peft loading:",max_token_id, model_vocab_size)
        datasets2 = load_dataset("text", data_files={"train": file_name, "test": file_name})
        trainer = SFTTrainer(
            model=model,
            train_dataset = datasets2["train"],
            dataset_text_field="text",
            eval_dataset = datasets2["test"],
            args=training_args
        )

    else:
        epochs=1
        if custom_block == True:
            special=f"custom_block_{custom_block}_{max_length}_date_{date}_path_gene_desc"
        else:
            special=f"custom_block_{custom_block}_path_gene_desc_date_{date}_v10"
        
        output_dir_name = f"./{m_name}_finetuned_{epochs}_epochs_{special}"

        training_args = TrainingArguments(
                output_dir= output_dir_name,
                num_train_epochs=epochs,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                fp16=True,
                gradient_accumulation_steps=8,
                weight_decay=0.01,
                logging_dir="./logs",
                gradient_checkpointing=True,
                optim="adafactor"
        )

        #print(datasets)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets,
            eval_dataset=datasets
        )



    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
        model = model.to("cuda")
    else:
        print("CUDA is not available. Training on CPU.")
        
    torch.cuda.empty_cache()
    trainer.train()
    
    trainer.save_model()
    tokenizer.save_pretrained(output_dir_name)
    print(output_dir_name)

    loaded_model = AutoModelForCausalLM.from_pretrained(output_dir_name)
    tokenizer = AutoTokenizer.from_pretrained(output_dir_name)
    max_token_id = len(tokenizer.get_vocab())
    model_vocab_size = loaded_model.config.vocab_size
    print("After saving:", max_token_id, model_vocab_size)



def main():
    m_name = "BioGPT-Large-PubMedQA"
    #example()
    tokenizer = AutoTokenizer.from_pretrained(m_name)

    #val 1: Pathway Nucleotide GPCRs contains ADORA1 with adenosine A1 receptor
    #val 2: Pathway: Prostaglandin synthesis and regulation | Genes: ['CYP11A1', 'ANXA5', 'EDNRB', 'PTGER4', 'S100A10', 'S100A6', 'PTGFR', 'PTGIS', 'PTGER2', 'PTGER1', 'HPGD', 'PTGS2', 'PRL', 'PLA2G4A', 'ANXA4', 'SCGB1A1', 'PTGDS', 'EDN1', 'ANXA3', 'HSD11B1', 'ANXA1', 'PTGDR', 'PTGIR', 'HSD11B2', 'PTGS1', 'ANXA6', 'EDNRA', 'TBXAS1', 'PTGER3', 'ANXA2', 'ABCC4', 'HPGDS', 'MITF', 'PTGES', 'PPARG', 'PPARGC1A', 'PPARGC1B', 'SOX9', 'AKR1B1', 'AKR1C3', 'CBR1', 'AKR1C1', 'AKR1C2', 'AKR1C1', 'TBXA2R', 'PTGFRN']
    #val 3: Pathway: Target of rapamycin signaling | Gene RPS6KB1
    #val 4: Pathway: 17q12 copy number variation syndrome | Gene: SLC35G3 | Gene description: solute carrier family 35 member G3 | Gene synonym name: TMEM21A
    #val 5_n_genes2: same as val 3 but without gene and with 2 genes per pathway
    #val 5_n_genes3: same as above but with 3 genes per pathway

    val = 4
    #val = "5_n_genes2"
    #val = "5_n_genes3"
    #file_name = f"/exports/sascstudent/svanderwal2/programs/test_new_models/training_data/custom_traintest_{val}.txt"

    file_name = "/exports/sascstudent/svanderwal2/programs/test_new_models/training_data/training_data_one_gene_multiple_pathways_18_04.txt"
    #file_name = "/exports/sascstudent/svanderwal2/programs/test_new_models/training_data/custom_traintest_1.txt"
    model = AutoModelForCausalLM.from_pretrained(m_name)

    lora = False
    custom_block = False
    expand_vocab = False
    datasets, model, tokenizer, max_length = preprocess_data(model, file_name, tokenizer, custom_block, expand_vocab)
    training_and_eval(datasets, model, m_name, tokenizer, file_name, lora, custom_block, max_length, expand_vocab, val)


main()