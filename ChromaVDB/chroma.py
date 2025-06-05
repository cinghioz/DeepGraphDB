import chromadb
from typing import Dict, List, Optional, Any, Union
import uuid
import time
from datetime import datetime
import pickle

class ChromaFramework:
    """
    A framework for managing records with CRUD operations using ChromaDB.
    Uses two separate collections: 'graph' for graph embeddings and 'text' for text embeddings.
    Records can exist in one or both collections with the same ID.
    Supports auto-generation of unique IDs and entity management.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the ChromaDB framework with two collections.
        
        Args:
            persist_directory: Directory to persist the database. If None, uses in-memory storage.
        """
        self.persist_directory = persist_directory or "./ChromaVDB"

        if persist_directory:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Create two collections
        self.graph_collection = self.client.get_or_create_collection(name="graph", configuration={ "hnsw": { "space": "cosine" } })
        self.text_collection = self.client.get_or_create_collection(name="text", configuration={ "hnsw": { "space": "cosine" } })
        
        self.collections = {
            "graph": self.graph_collection,
            "text": self.text_collection
        }

        try:
            with open(persist_directory+'/global_mapping.pkl', 'rb') as f:
                print("Found existing mapping file, loading...")
                self.global_to_vids_mapping = pickle.load(f)
        except FileNotFoundError:
            print("No mapping file found, starting with empty mapping")
            self.global_to_vids_mapping = {}

    def create_records(self, 
                    global_ids: List[int],
                    names: List[str],
                    entities: Optional[List[Optional[str]]] = [],
                    metadatas: Optional[List[Optional[Dict[str, Any]]]] = [],
                    documents: Optional[List[Optional[List[str]]]] = [],
                    embeddings: Optional[List[Optional[Dict[str, List[float]]]]] = []) -> List[str]:
        """
        Create multiple new records in the specified collection(s).
        
        Args:
            names: List of names for the records (mandatory)
            entities: Optional list of entities associated with each record (defaults to "default" if None)
            metadatas: Optional list of metadata dictionaries for each record
            documents: Optional list of document lists for each record
            embeddings: Optional list of embedding dictionaries for each record {"graph": [...], "text": [...]}
        
        Returns:
            List of record IDs (generated)
        """
        if not names:
            raise ValueError("At least one name must be provided")
        
        num_records = len(names)

        record_ids = [str(uuid.uuid4()) for _ in range(num_records)]
        
        # Check if any record already exists
        for record_id in record_ids:
            if self._record_exists(record_id):
                raise ValueError(f"Record with ID '{record_id}' already exists")
        
        # Prepare data structures for bulk operations
        collections_data = {"graph": {"ids": [], "documents": [], "metadatas": [], "embeddings": []},
                        "text": {"ids": [], "documents": [], "metadatas": [], "embeddings": []}}
        
        # Process each record
        for i in range(num_records):
            name = names[i]
            entity = entities[i] if entities[i] is not None else "default"
            metadata = metadatas[i] or {}
            document_list = documents[i]
            
            record_id = record_ids[i]
            
            # Prepare base metadata
            base_metadata = {
                "name": name,
                "entity": entity,
                "record_id": record_id
            }
            base_metadata.update(metadata)
            
            # Prepare document - use name as default if none provided
            record_document = document_list[0] if document_list else name

            if global_ids[i] in self.global_to_vids_mapping:
                raise ValueError(f"Global id {i} already exists in mapping!")
                
            self.global_to_vids_mapping[global_ids[i]] = record_id
            
            # Determine which collections to create records in
            if embeddings:
                # Create records in collections specified by embeddings dict
                for embedding_type, embedding_vector in embeddings.items():
                    if embedding_type not in self.collections:
                        raise ValueError(f"Invalid embedding type '{embedding_type}'. Must be 'graph' or 'text'")
                    
                    # Prepare metadata for this collection
                    record_metadata = base_metadata.copy()
                    record_metadata["embedding_type"] = embedding_type
                    
                    # Add to collection data
                    collections_data[embedding_type]["ids"].append(record_id)
                    collections_data[embedding_type]["documents"].append(record_document)
                    collections_data[embedding_type]["metadatas"].append(record_metadata)
                    collections_data[embedding_type]["embeddings"].append(embedding_vector[i].numpy())
            else:
                # If no embeddings provided, create in both collections with auto-generated embeddings
                for embedding_type in ["graph", "text"]:
                    record_metadata = base_metadata.copy()
                    record_metadata["embedding_type"] = embedding_type
                    
                    collections_data[embedding_type]["ids"].append(record_id)
                    collections_data[embedding_type]["documents"].append(record_document)
                    collections_data[embedding_type]["metadatas"].append(record_metadata)
                    # embeddings list will remain empty for auto-generation
        
        # Perform bulk operations on each collection
        created_collections = []
        try:
            for embedding_type, data in collections_data.items():
                if data["ids"]:  # Only process if there are records for this collection
                    collection = self.collections[embedding_type]
                    
                    if data["embeddings"]:
                        # Add with custom embeddings
                        collection.add(
                            ids=data["ids"],
                            documents=data["documents"],
                            metadatas=data["metadatas"],
                            embeddings=data["embeddings"]
                        )
                    else:
                        # Let ChromaDB auto-generate embeddings from documents
                        collection.add(
                            ids=data["ids"],
                            documents=data["documents"],
                            metadatas=data["metadatas"]
                        )
                    
                    created_collections.append((embedding_type, data["ids"]))
                    
        except Exception as e:
            # Cleanup: if any collection failed, try to remove from all collections that were created
            for embedding_type, ids_to_remove in created_collections:
                try:
                    self.collections[embedding_type].delete(ids=ids_to_remove)
                except Exception:
                    pass
            
            if "already exists" in str(e):
                raise ValueError(f"One or more records already exist")
            raise e

        with open(self.persist_directory+"/global_mapping.pkl", 'wb') as f:
            pickle.dump(self.global_to_vids_mapping, f)
        
        return record_ids
    
    def read_record(self, record_ids: List[str], include_embeddings: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Read a record by its ID from all collections where it exists.
        
        Args:
            record_id: The unique identifier of the record
            include_embeddings: Whether to include embeddings in the result
        
        Returns:
            Dictionary containing the record data or None if not found
        """
        found_collections = {}
        record_data = None
        records = []
        
        # Search in both collections
        for embedding_type, collection in self.collections.items():
            try:
                results = collection.get(
                    ids=record_ids,
                    include=["documents", "metadatas", "embeddings"] if include_embeddings else ["documents", "metadatas"]
                )
                
                i = 0
                for result in results:
                    if result['ids']:
                        metadata = result['metadatas'][0]
                    
                    # Initialize record_data on first find
                    if record_data is None:
                        record_data = {
                            "id": record_ids[i],
                            "name": metadata.get('name'),
                            "entity": metadata.get('entity', 'default'),
                            "collections": [],
                            "documents": [result['documents'][0]] if result['documents'] else [],
                            "metadata": {k: v for k, v in metadata.items() 
                                       if k not in ['name', 'entity', 'record_id', 'embedding_type', 'auto_counter']}
                        }
                        if include_embeddings:
                            record_data["embeddings"] = {}
                    
                    # Add collection info
                    record_data["collections"].append(embedding_type)
                    
                    if include_embeddings and "embeddings" in result:
                        record_data["embeddings"][embedding_type] = result['embeddings'][0]

                    i += 1
                    records.append(record_data)
                    
            except Exception:
                continue
        
        return records
    
    def update_record(self, 
                     record_id: str,
                     name: Optional[str] = None,
                     entity: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     documents: Optional[List[str]] = None,
                     embeddings: Optional[Dict[str, List[float]]] = None) -> bool:
        """
        Update an existing record in all collections where it exists.
        
        Args:
            record_id: The unique identifier of the record
            name: New name for the record
            entity: New entity for the record (defaults to "default" if explicitly set to None)
            metadata: New metadata (will merge with existing)
            documents: New documents (will replace existing)
            embeddings: New embeddings (will replace existing in specified collections)
        
        Returns:
            True if update was successful, False otherwise
        """
        # Find the record in collections
        existing_record = self.read_record(record_id, include_embeddings=True)
        if not existing_record:
            return False
        
        success = True
        
        # Update in each collection where the record exists
        for embedding_type in existing_record["collections"]:
            collection = self.collections[embedding_type]
            
            # Prepare updated data
            updated_metadata = existing_record["metadata"].copy()
            updated_name = name if name is not None else existing_record["name"]
            
            # Handle entity update - if explicitly set to None, use "default"
            if entity is not None:
                updated_entity = "default" if entity is None else entity
            else:
                updated_entity = existing_record["entity"]
            
            updated_metadata.update({
                "name": updated_name,
                "entity": updated_entity,
                "record_id": record_id,
                "embedding_type": embedding_type
            })
            
            if metadata:
                updated_metadata.update(metadata)
            
            updated_documents = documents[0] if documents else (existing_record["documents"][0] if existing_record["documents"] else updated_name)
            
            try:
                # Check if we have a new embedding for this collection
                if embeddings and embedding_type in embeddings:
                    # Update with new embedding
                    collection.update(
                        ids=[record_id],
                        documents=[updated_documents],
                        metadatas=[updated_metadata],
                        embeddings=[embeddings[embedding_type]]
                    )
                else:
                    # Update without changing embedding
                    collection.update(
                        ids=[record_id],
                        documents=[updated_documents],
                        metadatas=[updated_metadata]
                    )
                
            except Exception as e:
                print(f"Update error in {embedding_type} collection: {e}")
                success = False
        
        return success
    
    def delete_records(self, record_id: List[str]) -> bool:
        """
        Delete a record by its ID from any collection.
        
        Args:
            record_id: The unique identifier of the record
        
        Returns:
            True if deletion was successful, False otherwise
        """
        # Try to delete from both collections
        deleted = False
        for collection in self.collections.values():
            try:
                collection.delete(ids=record_id)
                deleted = True
            except Exception:
                continue
        
        if deleted:
            for global_id, rec_id in self.global_to_vids_mapping.items():
                if rec_id in record_id:
                    del self.global_to_vids_mapping[global_id]

            with open(self.persist_directory+"/global_mapping.pkl", 'wb') as f:
                pickle.dump(self.global_to_vids_mapping, f)

        return deleted
    
    def list_records(self, 
                    embedding_type: Optional[str] = None, 
                    entity: Optional[str] = None,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List records from specified collection(s).
        
        Args:
            embedding_type: Type of embedding ("graph", "text", or None for both)
            entity: Filter by entity (None for all entities)
            limit: Maximum number of records to return per collection
        
        Returns:
            List of record dictionaries
        """
        records = []
        
        # Determine which collections to search
        collections_to_search = {}
        if embedding_type and embedding_type in self.collections:
            collections_to_search[embedding_type] = self.collections[embedding_type]
        elif embedding_type is None:
            collections_to_search = self.collections
        else:
            return records  # Invalid embedding_type
        
        # Prepare where clause for entity filtering
        where_clause = {"entity": entity} if entity is not None else None
        
        # Get records from collections
        for emb_type, collection in collections_to_search.items():
            try:
                result = collection.get(
                    include=["documents", "metadatas", "embeddings"],
                    limit=limit,
                    where=where_clause
                )
                
                for i, record_id in enumerate(result['ids']):
                    metadata = result['metadatas'][i]
                    record_data = {
                        "id": record_id,
                        "name": metadata.get('name'),
                        "entity": metadata.get('entity', 'default'),
                        "embedding_type": emb_type,
                        "documents": [result['documents'][i]] if result['documents'] else [],
                        "metadata": {k: v for k, v in metadata.items() 
                                   if k not in ['name', 'entity', 'record_id', 'embedding_type', 'auto_counter']},
                        "embeddings": result['embeddings'][i] if 'embeddings' in result else None
                    }
                    records.append(record_data)
                    
            except Exception:
                continue
        
        return records
    
    # TODO: add filtering by docs
    def search_records(self, 
                      query: List[float] | str, 
                      embedding_type: str,
                      n_results: int = 5,
                      entity: Optional[str] = None,
                      where: Optional[Dict[str, Any]] = None,
                      where_docs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for records using similarity search in specified collection.
        
        Args:
            query: Text to search for | embsedding vector to search with
            embedding_type: Collection to search in ("graph" or "text")
            n_results: Maximum number of results to return
            entity: Filter by entity (None for all entities)
            where: Additional metadata filter conditions
            where_docs: Additional document filter conditions
        
        Returns:
            List of matching records with similarity scores
        """
        if embedding_type not in self.collections:
            return []
        
        collection = self.collections[embedding_type]
        
        # Prepare where clause
        where_clause = where.copy() if where else {}
        if entity is not None:
            where_clause["entity"] = entity
        
        try:
            if isinstance(query, str):
                result = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas", "distances", "embeddings"]
                )
            else:
                result = collection.query(
                    query_embeddings=query,
                    n_results=n_results,
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas", "distances", "embeddings"]
                )
            
            records = []
            for i, record_id in enumerate(result['ids'][0]):
                metadata = result['metadatas'][0][i]
                
                record_data = {
                    "id": record_id,
                    "name": metadata.get('name'),
                    "entity": metadata.get('entity', 'default'),
                    "embedding_type": embedding_type,
                    "documents": [result['documents'][0][i]] if result['documents'] else [],
                    "metadata": {k: v for k, v in metadata.items() 
                               if k not in ['name', 'entity', 'record_id', 'embedding_type', 'auto_counter']},
                    "embeddings": result['embeddings'][0][i] if 'embeddings' in result else None,
                    "distance": result['distances'][0][i]
                }
                records.append(record_data)
            
            return records
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get statistics about both collections.
        
        Returns:
            Dictionary with collection names and record counts
        """
        stats = {}
        for name, collection in self.collections.items():
            try:
                stats[name] = collection.count()
            except Exception:
                stats[name] = 0
        return stats
    
    def get_entities(self, embedding_type: Optional[str] = None) -> List[str]:
        """
        Get all unique entities from the specified collection(s).
        
        Args:
            embedding_type: Type of embedding ("graph", "text", or None for both)
        
        Returns:
            List of unique entity names
        """
        entities = set()
        
        # Determine which collections to search
        collections_to_search = {}
        if embedding_type and embedding_type in self.collections:
            collections_to_search[embedding_type] = self.collections[embedding_type]
        elif embedding_type is None:
            collections_to_search = self.collections
        else:
            return list(entities)
        
        # Get entities from collections
        for collection in collections_to_search.values():
            try:
                result = collection.get(include=["metadatas"])
                for metadata in result['metadatas']:
                    entity = metadata.get('entity', 'default')
                    entities.add(entity)
            except Exception:
                continue
        
        return sorted(list(entities))
    
    def _record_exists(self, record_id: str) -> bool:
        """Check if a record with the given ID already exists in any collection."""
        for collection in self.collections.values():
            try:
                result = collection.get(
                    ids=[record_id],
                    include=["metadatas"]
                )
                if result['ids']:
                    return True
            except Exception:
                continue
        return False