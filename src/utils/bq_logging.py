from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import logging 

class bq_logging:
    def __init__(self, project_id):
        self.client = bigquery.Client(project=project_id)
        logging.info(f"Initialized BigQuery Client for project {project_id}")

    def dataset_exists(self,dataset_id):
        try:
            self.client.get_dataset(dataset_id)  # Make an API request.
            logging.info("Dataset {} already exists".format(dataset_id))
            return True 
        except NotFound:
            logging.info("Dataset {} is not found".format(dataset_id))
            return False
        
    def create_dataset(self, dataset_id):
        dataset_ref = bigquery.DatasetReference.from_string(dataset_id, default_project=self.client.project)

        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "northamerica-northeast1"

        dataset = self.client.create_dataset(dataset, timeout=30)
        logging.info("Created dataset {}.{}".format(self.client.project, dataset.dataset_id))

    
    def table_exists(self, dataset_id, table_id):
        try:
            self.client.get_table(f"{dataset_id}.{table_id}")  # Make an API request.
            logging.info("Table {}:{} already exists.".format(dataset_id, table_id))
            return True
        except NotFound:
            logging.info("Table {}:{} is not found.".format(dataset_id, table_id))
            return False
        
    def create_log_table(self, dataset_id, table_id):
        schema = [
            bigquery.SchemaField("request_id", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("partition_dt", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("request", "JSON", mode="REQUIRED"),
            bigquery.SchemaField("response", "JSON", mode="REQUIRED"),
        ]
        
        table = bigquery.Table(f"{self.client.project}.{dataset_id}.{table_id}", schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
                                                        type_=bigquery.TimePartitioningType.DAY,
                                                        field="partition_dt",  # name of column to use for partitioning
                                                    )
        
        self.client.create_table(table)  # Make an API request.
        logging.info(
            "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
        )
    
    def insert_rows(self, table_id, row_data):
        errors = self.client.insert_rows_json(table_id, row_data)  # Make an API request.
        return errors



