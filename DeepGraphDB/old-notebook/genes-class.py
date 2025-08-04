# %%
import pandas as pd
import numpy as np
from ChromaVDB.chroma import ChromaFramework
from DeepGraphDB import DeepGraphDB
from tqdm import tqdm
import torch
import pickle

gdb = DeepGraphDB()
gdb.load_graph("/home/cc/PHD/dglframework/DeepKG/DeepGraphDB/graphs/primekg.bin")

vdb = ChromaFramework(persist_directory="./ChromaVDB/chroma_db")
records = vdb.list_records()

gene_embs = [record['embeddings'] for record in records if record['embedding_type'] == 'graph' and record['entity'] == 'geneprotein']
gene_names = [record['name'] for record in records if record['embedding_type'] == 'graph' and record['entity'] == 'geneprotein']

target = 35884

subg = gdb.get_k_hop_neighbors([target], k=2)

flat_nodes = []

for key, value in subg.items():
    flat_nodes.extend(value)

flat_nodes = list(set(flat_nodes)) 

gene_subg_names = [ gdb.node_data['geneprotein']['name'][gdb.global_to_local_mapping[fi][1]] for fi in flat_nodes \
                    if gdb.global_to_local_mapping[fi][0] == 'geneprotein' ] 

gene_embs = [record['embeddings'] for record in records if record['embedding_type'] == 'graph' and record['entity'] == 'geneprotein' \
             and record['name'] in gene_subg_names]

gene_names = [record['name'] for record in records if record['embedding_type'] == 'graph' and record['entity'] == 'geneprotein' \
              and record['name'] in gene_subg_names]

# %%
# gene_embs_ft = torch.load("/home/cc/PHD/dglframework/DeepKG/finetune-embs/geneprotein.pt")

#gene_embs.shape

# %%
import json
from typing import List, Dict, Tuple, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the output schema for LangChain's PydanticOutputParser
class ScoredEntity(BaseModel):
    """An entity from the biological knowledge graph with an assigned importance score."""
    name: str = Field(description="The name of the gene (e.g. TP53)")
    gene_class: int = Field(description="The number of the class from 0 to 3, relative to the gene classification.")

class EntityScoreOutput(BaseModel):
    """List of important entities and their scores."""
    important_entities: List[ScoredEntity] = Field(
        description="A list of genes with associated class"
    )

# Helper function to escape curly braces in a string
def escape_curly_braces(text: str) -> str:
    """Escapes single curly braces to double curly braces for f-string compatibility."""
    # Replace { with {{ and } with }}
    return text.replace("{", "{{").replace("}", "}}")

def gene_classification(genes_list: List[str], ollama_model_name: str = "alibayram/medgemma:27b") -> List[Dict[str, Any]]:

    # Initialize the Ollama LLM with structured output directly
    llm = ChatOllama(model=ollama_model_name).with_structured_output(EntityScoreOutput)
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant specialized in classifying genes based on their functional roles in the human body. "
                "For each gene, classify it according to the four functional categories defined below. Using the number of the class (0 to 3) to indicate the class of the gene. "
                "A single gene can belong to multiple categories if its functions apply. Provide the classification for each gene.\n"
                "1.  **Transcriptional & Epigenetic Regulators (class 0):** Identify genes whose products directly control gene expression."
                "includes proteins that bind to DNA to activate or repress transcription (transcription factors, "
                "co-activators, repressors) and enzymes that chemically modify histones or DNA to alter "
                "chromatin structure and accessibility (e.g., histone methyltransferases, acetyltransferases, "
                "demethylases). Also include core structural chromatin proteins like histones.\n"
                "Key Functions to Look For: Transcription factor activity, DNA binding, histone modification, "
                "chromatin remodeling, gene silencing, transcriptional activation/repression.\n"
                "2.  **Cell Cycle & Apoptosis Regulators (class 1):** Identify genes whose products function as critical control points in cell "
                "division or programmed cell death (apoptosis). This includes proteins that enforce cell cycle"
                "checkpoints (e.g., G1/S, G2/M), promote or inhibit proliferation (oncogenes, tumor "
                "suppressors), or are essential components of the apoptotic signaling cascade (e.g., death "
                "receptors, caspases, mitochondrial pathway members).\n"
                "Key Functions to Look For: Cell cycle arrest, apoptosis, programmed cell death, proliferation, tumor suppression, checkpoint control.\n"
                "3.  **Cell Signaling & Communication (class 2):** Identify genes whose products act as core components of signal"
                "transduction pathways. This includes cell surface receptors that bind to external ligands, "
                "intracellular kinases and phosphatases that relay signals through phosphorylation, and "
                "enzymes that generate, modify, or degrade signaling molecules to"
                "modulate information flow within or between cells.\n"
                "Key Functions to Look For: Signal transduction, kinase activity, receptor activity, G-protein signaling, NF-κB pathway, MAPK pathway, ubiquitin-editing."
                "4.  **Immune Development & Response (class 3):** Identify genes with a specialized role in the development, maturation, or "
                "function of the immune system. This includes genes that regulate the differentiation of "
                "immune cells (e.g., B-cells, T-cells), mediate the inflammatory response, control immune"
                "tolerance, or are directly involved in recognizing and eliminating pathogens or malignant cells.\n"
                "Key Functions to Look For: Lymphocyte differentiation, immune response, inflammation, cytokine signaling, T-cell/B-cell activation, negative regulation of immunity.\n"
                "**Output Format:**\n"
                "Provide your response as a JSON object, specifically as a list under the key 'important_entities'. "
                "Each item in the list should be a dictionary containing 'name' (the name of the gene)"
                "and its assigned 'gene_class'. Provide a class for ALL the genes provided in 'List of genes' (reply with a class from 0 to 3).\n"
            ),
            (
                "human",
                "**List of genes:**\n{genes_list}\n"
                "**Begin your classification.**"
            ),
        ]
    )

    # Create the LangChain chain
    chain = prompt | llm

    # Invoke the chain with the subgraph data
    response = chain.invoke({
        "genes_list": genes_list,
    })
    
    return response.important_entities

# %%
batch_size = 10

genes = gene_names
genes_class = []

for i in tqdm(range(0, len(genes), batch_size)):
    try:
        batch_genes = genes[i:i + batch_size]
    
        scored_entities = gene_classification(batch_genes, ollama_model_name="alibayram/medgemma:27b")

        genes_class.extend([entity.dict() for entity in scored_entities])
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")
        continue

with open("/home/cc/PHD/dglframework/DeepKG/gene_classes.pkl", "wb") as fp: 
    pickle.dump(genes_class, fp)
