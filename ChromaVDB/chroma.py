import chromadb
from typing import Dict, List, Optional, Any, Union
import uuid
import time
from datetime import datetime

#TODO: fixxare bulk load

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
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Create two collections
        self.graph_collection = self.client.get_or_create_collection(name="graph")
        self.text_collection = self.client.get_or_create_collection(name="text")
        
        self.collections = {
            "graph": self.graph_collection,
            "text": self.text_collection
        }

    def create_records(self, 
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
            embedding_dict = embeddings.keys()
            
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
            
            # Determine which collections to create records in
            if embedding_dict:
                # Create records in collections specified by embeddings dict
                for embedding_type, embedding_vector in embedding_dict.items():
                    if embedding_type not in self.collections:
                        raise ValueError(f"Invalid embedding type '{embedding_type}'. Must be 'graph' or 'text'")
                    
                    # Prepare metadata for this collection
                    record_metadata = base_metadata.copy()
                    record_metadata["embedding_type"] = embedding_type
                    
                    # Add to collection data
                    collections_data[embedding_type]["ids"].append(record_id)
                    collections_data[embedding_type]["documents"].append(record_document)
                    collections_data[embedding_type]["metadatas"].append(record_metadata)
                    collections_data[embedding_type]["embeddings"].append(embedding_vector)
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
        
        return record_ids
    
    def create_record(self, 
                     name: str,
                     entity: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     documents: Optional[List[str]] = None,
                     embeddings: Optional[Dict[str, List[float]]] = None) -> str:
        """
        Create a new record in the specified collection(s).
        
        Args:
            name: Name of the record (mandatory)
            entity: Entity associated with the record (defaults to "default" if None)
            metadata: Optional metadata dictionary
            documents: Optional list of text documents
            embeddings: Optional dictionary of embeddings {"graph": [...], "text": [...]}
        
        Returns:
            The record ID (generated or provided)
        """
        # Auto-generate ID if not provided
        record_id = str(uuid.uuid4())
        
        # Set default entity if None
        if entity is None:
            entity = "default"
        
        # Check if record already exists in any collection
        if self._record_exists(record_id):
            raise ValueError(f"Record with ID '{record_id}' already exists")
        
        # Prepare base record data
        base_metadata = {
            "name": name,
            "entity": entity,
            "record_id": record_id
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        # Prepare documents - use name as default document if none provided
        record_documents = documents[0] if documents else name
        
        # Determine which collections to create records in
        collections_to_create = []
        
        if embeddings:
            # Create records in collections specified by embeddings dict
            for embedding_type, embedding_vector in embeddings.items():
                if embedding_type in self.collections:
                    collections_to_create.append((embedding_type, embedding_vector))
                else:
                    raise ValueError(f"Invalid embedding type '{embedding_type}'. Must be 'graph' or 'text'")
        else:
            # If no embeddings provided, create in both collections with auto-generated embeddings
            collections_to_create = [("graph", None), ("text", None)]
        
        # Create records in specified collections
        try:
            for embedding_type, embedding_vector in collections_to_create:
                collection = self.collections[embedding_type]
                
                # Prepare metadata for this collection
                record_metadata = base_metadata.copy()
                record_metadata["embedding_type"] = embedding_type
                
                if embedding_vector:
                    # Add with custom embedding
                    collection.add(
                        ids=[record_id],
                        documents=[record_documents],
                        metadatas=[record_metadata],
                        embeddings=[embedding_vector]
                    )
                else:
                    # Let ChromaDB auto-generate embedding from document
                    collection.add(
                        ids=[record_id],
                        documents=[record_documents],
                        metadatas=[record_metadata]
                    )
                    
        except Exception as e:
            # Cleanup: if any collection failed, try to remove from all collections
            for embedding_type, _ in collections_to_create:
                try:
                    self.collections[embedding_type].delete(ids=[record_id])
                except Exception:
                    pass
            
            if "already exists" in str(e):
                raise ValueError(f"Record with ID '{record_id}' already exists")
            raise e
        
        return record_id
    
    def read_record(self, record_id: str, include_embeddings: bool = False) -> Optional[Dict[str, Any]]:
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
        
        # Search in both collections
        for embedding_type, collection in self.collections.items():
            try:
                result = collection.get(
                    ids=[record_id],
                    include=["documents", "metadatas", "embeddings"] if include_embeddings else ["documents", "metadatas"]
                )
                
                if result['ids']:
                    metadata = result['metadatas'][0]
                    
                    # Initialize record_data on first find
                    if record_data is None:
                        record_data = {
                            "id": record_id,
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
                    
            except Exception:
                continue
        
        return record_data
    
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
    
    def delete_record(self, record_id: str) -> bool:
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
                collection.delete(ids=[record_id])
                deleted = True
            except Exception:
                continue
        
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
    
    def search_records(self, 
                      query_text: str, 
                      embedding_type: str,
                      n_results: int = 5,
                      entity: Optional[str] = None,
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for records using similarity search in specified collection.
        
        Args:
            query_text: Text to search for
            embedding_type: Collection to search in ("graph" or "text")
            n_results: Maximum number of results to return
            entity: Filter by entity (None for all entities)
            where: Additional metadata filter conditions
        
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
            result = collection.query(
                query_texts=[query_text],
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