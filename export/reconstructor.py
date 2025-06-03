import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.schema import CreateTable


class DatabaseReconstructor:
    """
    Reconstructs a database with synthetic data
    """

    def __init__(self, schema, relationships=None):
        self.schema = schema
        self.relationships = relationships or []
        self.sql_schema = {}

    def generate_sql_schema(self, dialect='sqlite') -> Dict[str, str]:
        """
        Generate SQL schema for tables

        Parameters:
        dialect: SQL dialect ('sqlite', 'postgresql', 'mysql')

        Returns dictionary with table creation SQL
        """
        # Create SQLAlchemy metadata
        metadata = MetaData()

        # Create tables
        tables = {}
        for table_name, table_schema in self.schema.items():
            columns = []

            # Add primary key
            columns.append(Column('id', Integer, primary_key=True))

            # Add columns based on schema
            for column_name, column_info in table_schema.get('columns', {}).items():
                col_type = column_info.get('type')

                if col_type == 'numeric':
                    col = Column(column_name, Float)
                elif col_type == 'categorical':
                    col = Column(column_name, String)
                elif col_type == 'datetime':
                    col = Column(column_name, DateTime)
                else:
                    col = Column(column_name, String)

                columns.append(col)

            # Create table
            tables[table_name] = Table(table_name, metadata, *columns)

        # Add foreign key relationships
        for relationship in self.relationships:
            parent_table = relationship.get('parent_table')
            child_table = relationship.get('child_table')
            parent_column = relationship.get('parent_column')
            child_column = relationship.get('child_column')

            if (parent_table in tables and child_table in tables and
                    parent_column and child_column):
                # Add foreign key column to child table
                tables[child_table].append_column(
                    Column(
                        child_column, Integer,
                        ForeignKey(f"{parent_table}.{parent_column}")
                    )
                )

        # Generate SQL schema
        sql_schema = {}
        for table_name, table in tables.items():
            sql_schema[table_name] = str(CreateTable(table).compile(
                dialect=create_engine(f"{dialect}://").dialect
            ))

        # Store schema
        self.sql_schema = sql_schema

        return sql_schema

    def reconstruct_sqlite_database(self, data_dict: Dict[str, pd.DataFrame],
                                    db_path: str) -> bool:
        """
        Reconstruct SQLite database with synthetic data

        Parameters:
        data_dict: Dictionary of {table_name: dataframe}
        db_path: Path to output SQLite database

        Returns True if successful
        """
        try:
            # Generate SQL schema if not already done
            if not self.sql_schema:
                self.generate_sql_schema(dialect='sqlite')

            # Create connection to SQLite database
            conn = sqlite3.connect(db_path)

            # Create tables and insert data
            for table_name, df in data_dict.items():
                if table_name in self.sql_schema:
                    # Create table
                    conn.execute(self.sql_schema[table_name])

                    # Insert data
                    df.to_sql(table_name, conn, if_exists='append', index=False)

            # Commit changes and close connection
            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"Error reconstructing database: {e}")
            return False

    def reconstruct_sql_database(self, data_dict: Dict[str, pd.DataFrame],
                                 connection_string: str) -> bool:
        """
        Reconstruct database with synthetic data using SQLAlchemy

        Parameters:
        data_dict: Dictionary of {table_name: dataframe}
        connection_string: SQLAlchemy connection string

        Returns True if successful
        """
        try:
            # Create engine
            engine = create_engine(connection_string)

            # Generate SQL schema if not already done
            dialect = connection_string.split('://')[0]
            if not self.sql_schema:
                self.generate_sql_schema(dialect=dialect)

            # Create tables and insert data
            for table_name, df in data_dict.items():
                if table_name in self.sql_schema:
                    # Create table
                    engine.execute(self.sql_schema[table_name])

                    # Insert data
                    df.to_sql(table_name, engine, if_exists='append', index=False)

            return True

        except Exception as e:
            print(f"Error reconstructing database: {e}")
            return False

    def reconstruct_csv_files(self, data_dict: Dict[str, pd.DataFrame],
                              output_dir: str) -> Dict[str, str]:
        """
        Export synthetic data to CSV files

        Parameters:
        data_dict: Dictionary of {table_name: dataframe}
        output_dir: Directory to output CSV files

        Returns dictionary with {table_name: file_path}
        """
        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Export each table to CSV
        file_paths = {}
        for table_name, df in data_dict.items():
            file_path = os.path.join(output_dir, f"{table_name}.csv")
            df.to_csv(file_path, index=False)
            file_paths[table_name] = file_path

        return file_paths