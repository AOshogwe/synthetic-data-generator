from sqlalchemy import create_engine
import pandas as pd


class DatabaseConnector:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)

    def get_tables(self):
        """Return list of tables in database"""
        return self.engine.table_names()

    def get_schema(self, table_name):
        """Retrieve schema information for a table"""
        # Implementation

    def sample_data(self, table_name, sample_size=10000):
        """Sample data from a table"""
        query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
        return pd.read_sql(query, self.engine)

    def get_relationships(self):
        """Discover foreign key relationships between tables"""
        # Implementation