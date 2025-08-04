# %%
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate

import pandas as pd
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# These classes are assumed to be defined in your environment
from ChromaVDB.chroma import ChromaFramework
from DeepGraphDB import DeepGraphDB

gdb = DeepGraphDB()
gdb.load_graph("/home/cc/PHD/dglframework/DeepKG/DeepGraphDB/graphs/primekg.bin")
# vdb = ChromaFramework(persist_directory="./ChromaVDB/chroma_db")
# records = vdb.list_records()
# names = [record['name'] for record in records if record['embedding_type'] == 'graph']
# "alibayram/medgemma:27b"
# "35884" # Diffuse b-cell lymphoma in graph

# %%
import json
from typing import List, Dict, Tuple, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the output schema for LangChain's PydanticOutputParser
class ScoredEntity(BaseModel):
    """An entity from the biological knowledge graph with an assigned importance score."""
    entity: str = Field(description="The entity in 'entity_type: name' format (e.g., disease: B-cell lymphoma)")
    score: int = Field(description="Importance score from 2 to 5, relative to the source entity. 1 is default for unlisted entities.")

class EntityScoreOutput(BaseModel):
    """List of important entities and their scores."""
    important_entities: List[ScoredEntity] = Field(
        description="A list of entities from the subgraph with an importance score greater than 1."
    )

# Helper function to escape curly braces in a string
def escape_curly_braces(text: str) -> str:
    """Escapes single curly braces to double curly braces for f-string compatibility."""
    # Replace { with {{ and } with }}
    return text.replace("{", "{{").replace("}", "}}")

def score_subgraph_entities(
    source_entity: str,
    k_hops: int,
    # subgraph_triplets: List[Tuple[str, str, str]],
    subgraph_triplets: List[str],
    ollama_model_name: str = "alibayram/medgemma:27b"
) -> List[Dict[str, Any]]:
    """
    Uses MedGemma via Ollama and LangChain to score entities in a k-hop subgraph.

    Args:
        source_entity (str): The source entity (e.g., "disease: B-cell lymphoma").
        k_hops (int): The number of hops for the subgraph extraction.
        subgraph_triplets (List[Tuple[str, str, str]]): List of (head, relation, tail) triplets.
        meta_paths (List[str]): List of descriptions of meta-paths in the subgraph.
        ollama_model_name (str): The name of the Ollama model to use (e.g., "medgemma:27b").

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each with 'entity' and 'score' for important entities.
    """

    # Initialize the Ollama LLM with structured output directly
    llm = ChatOllama(model=ollama_model_name).with_structured_output(EntityScoreOutput)
    
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in biological knowledge graphs and entity relationship analysis. "
                "Your task is to identify and score the most important entities within a k-hop subgraph, "
                "relative to a given source entity. The importance score should reflect how strongly and "
                "directly an entity is related to the source entity, and its potential biological "
                "significance in that context. \n"
                "The entities are provided in the format entity_type: entity_name "
                "\n\n"
                "**Instructions for Scoring:**\n"
                # "1. **Analyze Relationships:** Carefully examine the provided entities. Also consider how meta-paths highlight indirect but potentially significant connections.\n"
                "1. **Analyze Relationships:** Carefully examine the provided entities.\n"
                "2. **Identify Importance:** Focus on entities that have a direct and strong biological connection to the Source Entity, or those that are central to multiple important relationships within the subgraph. Entities directly connected in the first hop are generally more important.\n"
                "3. **Scoring Scale (1-5):**\n"
                "   * **Score 5 (Highly Important):** Entities directly and strongly related to the Source Entity, representing core biological associations. These are often primary targets, key regulators, or direct causes/treatments.\n"
                "   * **Score 4 (Very Important):** Entities with strong, direct, and highly relevant connections, or those central to significant meta-paths from the Source Entity.\n"
                "   * **Score 3 (Moderately Important):** Entities with clear, but perhaps less direct or less universally critical connections. They still offer valuable biological insights.\n"
                "   * **Score 2 (Slightly Important):** Entities with indirect or less prominent connections, but still part of the subgraph and potentially relevant in a broader context.\n"
                "   * **Score 1 (Least Important / Baseline):** All other entities within the subgraph that are not explicitly assigned a higher score, or entities whose relevance is minimal in the context of the Source Entity.\n\n"
                "**Output Format:**\n"
                "Provide your response as a JSON object, specifically as a list under the key 'important_entities'. "
                "Each item in the list should be a dictionary containing 'entity' (in 'entity_type: name' format) "
                "and its assigned 'score'. Provide a score for ALL the entities provided in 'Source Entity' (reply with score 1 if there are irrelevant entities).\n"
                # "Provide your response as a JSON object."
                # "Each item in the list should be a dictionary containing 'entity'"
                # "and its assigned 'score' (eg: 'tk53': 4). Only include entities with a score greater than 1. All entities not "
                # "listed explicitly in your output are implicitly considered to have a score of 1.\n"
            ),
            ("human",
             "**Source Entity:**\n{source_entity}\n\n"
             "**Subgraph Entities:**\n{subgraph_entities}\n\n"
             "**Begin your analysis.**"
            ),
        ]
    )

    # Create the LangChain chain
    chain = prompt | llm

    # Invoke the chain with the subgraph data
    response = chain.invoke({
        "source_entity": source_entity,
        "subgraph_entities": subgraph_triplets,
    })
    
    # 'response' is an instance of EntityScoreOutput
    # You return response.important_entities, which is a List[ScoredEntity]
    return response.important_entities

# %%
def get_k_hop_subgraph(gid: int, k: int = 2):
    subg = gdb.get_k_hop_neighbors([gid], k=k)

    flat_nodes = []

    for key, value in subg.items():
        if int(key) > 0:
            flat_nodes.extend(value)
        # if int(key) > 0 and int(key) == 1:
        #     flat_nodes.extend(value)

        # if int(key) > 0 and int(key) > 1:
        #     flat_nodes.extend(list(value)[:500])

    flat_nodes = list(set(flat_nodes)) 

    return [ gdb.global_to_local_mapping[node][0]+": "+gdb.node_data[gdb.global_to_local_mapping[node][0]]['name'][gdb.global_to_local_mapping[node][1]] for node in flat_nodes ]

# %%
source_entity = "disease: B-cell lymphoma"
source_entity_gid = 35884
k_hops = 2
triplets = get_k_hop_subgraph(source_entity_gid, k=k_hops)

print(f"\n--- Scoring {len(triplets)} nodes for {source_entity} ({k_hops}-hop) ---")

conditioning = []
batch_size = 25

for i in tqdm(range(0, len(triplets), batch_size)):
    try:
        batch = score_subgraph_entities(source_entity, k_hops, triplets[i:i + batch_size])
        conditioning.extend([entity.dict() for entity in batch])
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {e}")
        continue


# scored_entities = score_subgraph_entities(source_entity, k_hops, triplets[:10])
# json_serializable_entities = [entity.dict() for entity in scored_entities]
# print(json.dumps(json_serializable_entities, indent=2))

# %%

with open("/home/cc/PHD/dglframework/DeepKG/conditioning.pkl", "wb") as f:
    pickle.dump(conditioning, f)


