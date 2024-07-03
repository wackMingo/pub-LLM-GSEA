def part_two(conv_dic, vocab_dic):
    new_dic = {}
    wikipathways= "wikipathways.gmt"

    vocab = []
    with open(wikipathways, "r") as wiki:
        for line in wiki.readlines():
            new_line = []
            line = line.split("\t")
            pathway = line[0].split("%")[0]
            vocab.append(pathway)
            for gene in line[2:]:
                gene = gene.strip("\n")
                conv_gene = conv_dic.get(gene)

                vocab_gene = vocab_dic.get(gene)
                if vocab_gene == '' or vocab_gene ==None:
                    vocab_gene = gene
                for x in vocab_gene:
                    vocab.append(x)

                if conv_gene == '':
                    conv_gene = gene
                new_line.append(conv_gene)
            new_dic[pathway] = new_line

    #remove duplicates
    file_vocab = open("make_custom_vocab.txt", "w")
    vocab = set(vocab)
    for x in vocab:
        file_vocab.write(x+"\n")
    file_vocab.close()
    return new_dic

def part_two_2(vocab_dic, val_conv):
    new_dic = {}
    wikipathways= "wikipathways.gmt"
    #print(vocab_dic)

    with open(wikipathways, "r") as wiki:
        for line in wiki.readlines():
            new_line = []
            line = line.split("\t")
            #this line contains the pathway with all the genes in them.
            pathway = line[0].split("%")[0]
            for gene in line[2:]:
                #convert all the genes from the pathway in to good type using the vocab dic where data is gotten from biomart
                gene = gene.strip("\n")
                conv_gene = vocab_dic.get(gene)
                if conv_gene == '' or conv_gene == None:
                    conv_gene = gene
                else:
                #conv_gene[0] is description
                #etc., look at read2
                #if we want to add extra text to every gene change the line beneath here
                    if val_conv == 0:
                        conv_gene = conv_gene[1]
                    elif val_conv == 1:
                        conv_gene = [conv_gene[1], conv_gene[0], conv_gene[2]]
                new_line.append(conv_gene)
            new_dic[pathway] = new_line
    return new_dic

def read():
    conv_dic = {}
    vocab_dic = {}
    biomart = "mart_export_updated.txt"
    with open(biomart, "r") as mart:
        for line in mart:
            line = line.split("\t")
            #conv_dic[line[5].strip("\n")] = f"is a pathway which contains the following genes, every new gene description is a new gene: Gene description: {line[0]}, Gene name: {line[1]}, Gene synonym: {line[2]}, Phenotype description: {line[3]}"
            zero = line[0].split(" [")[0]
            conv_dic[line[5].strip("\n")] = f"{line[1]} with {zero}"

            vocab_dic[line[5].strip("\n")] = [line[1], zero, line[2]]
            #NCBI ID: Gene description, gene name, gene synonym, transcript name, phenotype description
    return conv_dic, vocab_dic
    
def read2():
    vocab_dic = {}
    biomart = "mart_export_updated.txt"
    with open(biomart, "r") as mart:
        for line in mart:
            line = line.split("\t")
            #line[0] = gene description
            #line[1] = gene name
            #line[2] = gene synonym
            #line[3] = phenotype description
            #line[4] = transcript name
            #line[5] = gene
            zero = line[0].split(" [")[0]
            vocab_dic[line[5].strip("\n")] = [zero, line[1], line[2], line[3], line[4]]

    #conv dic:
    #key: 57002(gene), value: gene {text} description
    #vocab_dic:
    #key: number(gene), value: gene name, description, gene synonym

    return vocab_dic
            
                
def write(new_dic):
    file = open("custom_traintest_1.txt", "w")
    for pathway in new_dic:
        for gene in new_dic[pathway]:
            file.write(f"Pathway {pathway} contains {gene}\n")
    file.close()
        
def write2(new_dic):
    file = open("custom_traintest_2.txt", "w")
    for pathway in new_dic:
        file.write(f"Pathway: {pathway} | Genes: {new_dic[pathway]}\n")
    file.close()

def write3(new_dic):
    file = open("custom_traintest_3.txt", "w")
    for pathway in new_dic:
        for gene in new_dic[pathway]:
            file.write(f"Pathway: {pathway} | Gene {gene}\n")
    file.close()

def write4(new_dic):
    file = open("custom_traintest_4.txt", "w")
    for pathway in new_dic:
        print(new_dic[pathway])
        for gene in new_dic[pathway]:
            file.write(f"Pathway: {pathway} | Gene: {gene[0]} | Gene description: {gene[1]} | Gene synonym name: {gene[2]}\n")
    file.close()

def write5(new_dic):
    skip = 3
    file = open(f"custom_traintest_5_n_genes{skip}.txt", "w")
    for pathway in new_dic:
        for gene in range(0,len(new_dic[pathway])-1,skip):
            if len(new_dic[pathway])-gene > 2:
                file.write(f"\nPathway: {pathway} | ")
                for x in range(skip):
                    file.write(f"{new_dic[pathway][gene+x][0]} ")
        if len(new_dic[pathway])%2 == 1 and len(new_dic[pathway]) != 1 and skip % 2 == 0:
            file.write(f"{new_dic[pathway][gene+2][0]} ") 
    file.close()

def recombination(new_dic):
    gene_pathways_dic = {}
    for pathway in new_dic:
        gene_list = new_dic[pathway]
        pathway = pathway.strip().strip("\n")
        for gene in gene_list:
            if gene_pathways_dic.get(gene) == None:
                gene_pathways_dic[gene] = [pathway]
            else:
                gene_pathways_dic[gene].append(pathway)
    
    file = open("training_data_one_gene_multiple_pathways_18_04.txt", "w")
    for gene in gene_pathways_dic:
        file.write("\n"+gene+": ")
        for pathway in set(gene_pathways_dic[gene]):
            file.write(pathway+", ")
    file.close()






def main():
    val=5
    #val 1: Pathway Nucleotide GPCRs contains ADORA1 with adenosine A1 receptor
    #val 2: Pathway: Prostaglandin synthesis and regulation | Genes: ['CYP11A1', 'ANXA5', 'EDNRB', 'PTGER4', 'S100A10', 'S100A6', 'PTGFR', 'PTGIS', 'PTGER2', 'PTGER1', 'HPGD', 'PTGS2', 'PRL', 'PLA2G4A', 'ANXA4', 'SCGB1A1', 'PTGDS', 'EDN1', 'ANXA3', 'HSD11B1', 'ANXA1', 'PTGDR', 'PTGIR', 'HSD11B2', 'PTGS1', 'ANXA6', 'EDNRA', 'TBXAS1', 'PTGER3', 'ANXA2', 'ABCC4', 'HPGDS', 'MITF', 'PTGES', 'PPARG', 'PPARGC1A', 'PPARGC1B', 'SOX9', 'AKR1B1', 'AKR1C3', 'CBR1', 'AKR1C1', 'AKR1C2', 'AKR1C1', 'TBXA2R', 'PTGFRN']
    #val 3: Pathway: Target of rapamycin signaling | Gene RPS6KB1
    #val 4: Pathway: 17q12 copy number variation syndrome | Gene: SLC35G3 | Gene description: solute carrier family 35 member G3 | Gene synonym name: TMEM21A
    #val 5_2: 1 pathway two genes
    #val 5_3: 1 pathway three genes
    pathway_genes = False
    
    if pathway_genes == True:
        if val == 1:
            #run this if you want to make custom vocab, need to make better system to make vocab to expand vocab
            conv_dic, vocab_dic = read()
            new_dic = part_two(conv_dic, vocab_dic)
            write(new_dic)

        if val == 2:
            vocab_dic = read2()
            val_conv = 0
            new_dic = part_two_2(vocab_dic, val_conv)
            write2(new_dic)

        if val == 3:
            vocab_dic = read2()
            val_conv = 0
            new_dic = part_two_2(vocab_dic, val_conv)
            write3(new_dic)

        if val == 4:
            vocab_dic = read2()
            val_conv = 1
            new_dic = part_two_2(vocab_dic, val_conv)
            write4(new_dic)

        if val == 5:
            vocab_dic = read2()
            val_conv = 1
            new_dic = part_two_2(vocab_dic, val_conv)
            write5(new_dic)

    else:
        vocab_dic = read2()
        #print(vocab_dic)
        val_conv = 0
        new_dic = part_two_2(vocab_dic, val_conv)
        recombination(new_dic)
    

    #make like 80/20 split



main()
