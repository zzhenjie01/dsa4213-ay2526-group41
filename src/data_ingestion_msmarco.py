import weaviate
import weaviate.classes.config as wvc
import json
import time
import os
import logging
import sys

# -----------------------------------------------
# Configure Script to be Current Directory
# -----------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Current working directory set to: {os.getcwd()}")

# -----------------------------------------------
# Configure Logging
# -----------------------------------------------
LOG_DIR = os.path.join(os.getcwd(), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
LOG_PATH = os.path.join(LOG_DIR, f"weaviate_ingestion_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    client = None  # Define client in outer scope to access in 'finally'
    try:
        # -----------------------------------------------
        # Connect to Weaviate
        # -----------------------------------------------
        HOST = "127.0.0.1"
        PORT = 8080
        GRPC_PORT = 50051

        try:
            # Connect to Weviate instance without authentication
            client = weaviate.connect_to_local(host=HOST, port=PORT, grpc_port=GRPC_PORT)
            logger.info("Connected to Weaviate successfully.")
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            sys.exit(1)
            
        # -----------------------------------------------
        # Define and Create Schema ("Class")
        # -----------------------------------------------
        COLLECTION_NAME = "MS_MARCO"

        # If the collection already exists, delete it
        if client.collections.exists(COLLECTION_NAME):
            logger.info(f"Collection '{COLLECTION_NAME}' already exists. Deleting it.")
            client.collections.delete(COLLECTION_NAME)
            time.sleep(2)  # wait for deletion to propagate

        # Create collection with v4 API
        logger.info(f"Creating collection '{COLLECTION_NAME}' in Weaviate schema.")
        client.collections.create(
            name=COLLECTION_NAME,
            # `vectorizer_config` is deprecated, use `vector_config` instead
            # Tell weviate we will provide our own embeddings
            vector_config=wvc.Configure.Vectors.self_provided(),
            # BM25 / Inverted Index configuration
            inverted_index_config=wvc.Configure.inverted_index(
                bm25_b=0.75, # standard BM25 parameters
                bm25_k1=1.2, # standard BM25 parameters
                stopwords_preset=wvc.StopwordsPreset.EN
            ),
            # Define properties of our data fields
            properties=[
                wvc.Property(
                    name="pid",
                    data_type=wvc.DataType.TEXT,
                    description="The unique identifier for each passage.",
                    tokenization=wvc.Tokenization.FIELD, # important for exact matching
                    index_searchable=True
                ),
                wvc.Property(
                    name="passage",
                    data_type=wvc.DataType.TEXT,
                    description="The text content of the passage.",
                    tokenization=wvc.Tokenization.WORD, # enable full-text search (BM25)
                    index_searchable=True
                )
            ]
        )

        # -----------------------------------------------
        # Read and Ingest Passage Data in Batches
        # -----------------------------------------------
        PASSAGES_WITH_EMBEDDINGS_PATH = "../data/ms_marco/processed/passages_with_embeddings.jsonl"
        LOG_EVERY_N = 10000 # Log progress every N passages

        # Specifiy the collection to ingest into
        collection = client.collections.use(COLLECTION_NAME)

        total_ingested = 0
        start_time = time.time()
        logger.info(f"Starting data ingestion from {PASSAGES_WITH_EMBEDDINGS_PATH}...")

        # The `with` block ensures all remaining objects are sent
        # Adjust the batch size and concurrency connections as needed
        with collection.batch.fixed_size(batch_size=2000, concurrent_requests=6) as batch:
            try:
                with open(PASSAGES_WITH_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue  # skip empty lines

                        try:
                            # Parse JSON line
                            # The properties "pid" and "passage" must match the schema
                            # {"pid": "0", "passage": "The presence of communication...", "embedding": [0.01, -0.45, ..., 0.23]}
                            rec = json.loads(line)
                            data_object = {
                                "pid": rec["pid"],
                                "passage": rec["passage"]
                            }
                            passage_vector = rec["embedding"]
                            # Add the data object and vectorto the batch
                            batch.add_object(
                                properties=data_object, 
                                vector=passage_vector)
                            total_ingested += 1

                            if total_ingested % LOG_EVERY_N == 0:
                                logger.info(f"Total {total_ingested} passages ingested so far.")
                        
                        except json.JSONDecodeError as jde:
                            logger.error(f"JSON decode error for line: {line.strip()}. Error: {jde}")
                            continue
                        except TypeError as te:
                            logger.error(f"Type error for record: {rec}. Error: {te}")
                            continue

            except FileNotFoundError as fnfe:
                logger.error(f"Passage collection file not found: {fnfe}")
                sys.exit(1)

        # -----------------------------------------------
        # Post-Ingestion Reporting and Cleanup
        # -----------------------------------------------
        end_time = time.time()

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            logger.error(f"Number of failed imports: {len(failed_objects)}")
            logger.error(f"First failed objects: {failed_objects[0]}")
        elif not failed_objects:
            logger.info(f"Data ingestion completed. Total passages ingested: {total_ingested}")
            logger.info(f"Total time taken: {(end_time - start_time):.2f} seconds.")

    except KeyboardInterrupt:
        logger.info("Data ingestion interrupted by user. Shutting down gracefully...")

    finally:
        # Ensure client always closed
        if client:
            client.close()
            logger.info("Weaviate client connection closed.")
        logger.info("Script finished.")

if __name__ == "__main__":
    main()