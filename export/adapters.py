import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, BinaryIO
import json
import os
import zipfile
import io
from datetime import datetime


class ExportAdapter:
    """Base adapter for exporting synthetic data"""

    @classmethod
    def create_adapter(cls, format_type: str):
        """
        Factory method to create an appropriate export adapter based on format
        """
        if format_type.lower() == 'csv':
            return CSVExportAdapter()
        elif format_type.lower() == 'json':
            return JSONExportAdapter()
        elif format_type.lower() == 'sql':
            return SQLExportAdapter()
        elif format_type.lower() == 'parquet':
            return ParquetExportAdapter()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def export(self, df, output_path: str) -> bool:
        """
        Export the dataframe to the specified path
        Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement export method")


class CSVExportAdapter(ExportAdapter):
    """
    Exports data to CSV format
    """

    def export(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
               output_path: str) -> bool:
        """
        Export data to CSV format

        Parameters:
        data: DataFrame or dictionary of DataFrames
        output_path: Path to output file or directory

        Returns True if successful
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Single dataframe
                data.to_csv(output_path, index=False)
            else:
                # Dictionary of dataframes
                os.makedirs(output_path, exist_ok=True)

                for table_name, df in data.items():
                    file_path = os.path.join(output_path, f"{table_name}.csv")
                    df.to_csv(file_path, index=False)

            return True

        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False


class JSONExportAdapter(ExportAdapter):
    """
    Exports data to JSON format
    """

    def export(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
               output_path: str) -> bool:
        """
        Export data to JSON format

        Parameters:
        data: DataFrame or dictionary of DataFrames
        output_path: Path to output file or directory

        Returns True if successful
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Single dataframe
                data.to_json(output_path, orient='records', date_format='iso')
            else:
                # Dictionary of dataframes
                if output_path.endswith('.json'):
                    # Export as a single JSON file with tables as keys
                    json_data = {
                        table_name: df.to_dict(orient='records')
                        for table_name, df in data.items()
                    }

                    with open(output_path, 'w') as f:
                        json.dump(json_data, f, default=self._json_serializer)
                else:
                    # Export as multiple JSON files in a directory
                    os.makedirs(output_path, exist_ok=True)

                    for table_name, df in data.items():
                        file_path = os.path.join(output_path, f"{table_name}.json")
                        df.to_json(file_path, orient='records', date_format='iso')

            return True

        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False

    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()

        raise TypeError(f"Type {type(obj)} not serializable")


class ParquetExportAdapter(ExportAdapter):
    """
    Exports data to Parquet format
    """

    def export(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
               output_path: str) -> bool:
        """
        Export data to Parquet format

        Parameters:
        data: DataFrame or dictionary of DataFrames
        output_path: Path to output file or directory

        Returns True if successful
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Single dataframe
                data.to_parquet(output_path, index=False)
            else:
                # Dictionary of dataframes
                os.makedirs(output_path, exist_ok=True)

                for table_name, df in data.items():
                    file_path = os.path.join(output_path, f"{table_name}.parquet")
                    df.to_parquet(file_path, index=False)

            return True

        except Exception as e:
            print(f"Error exporting to Parquet: {e}")
            return False


class SQLiteExportAdapter(ExportAdapter):
    """
    Exports data to SQLite database
    """

    def export(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
               output_path: str) -> bool:
        """
        Export data to SQLite database

        Parameters:
        data: DataFrame or dictionary of DataFrames
        output_path: Path to output SQLite database file

        Returns True if successful
        """
        try:
            import sqlite3

            # Create connection to SQLite database
            conn = sqlite3.connect(output_path)

            if isinstance(data, pd.DataFrame):
                # Single dataframe
                table_name = os.path.splitext(os.path.basename(output_path))[0]
                data.to_sql(table_name, conn, if_exists='replace', index=False)
            else:
                # Dictionary of dataframes
                for table_name, df in data.items():
                    df.to_sql(table_name, conn, if_exists='replace', index=False)

            # Commit changes and close connection
            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"Error exporting to SQLite: {e}")
            return False


class ExcelExportAdapter(ExportAdapter):
    """
    Exports data to Excel format
    """

    def export(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
               output_path: str) -> bool:
        """
        Export data to Excel format

        Parameters:
        data: DataFrame or dictionary of DataFrames
        output_path: Path to output Excel file

        Returns True if successful
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Single dataframe
                data.to_excel(output_path, index=False)
            else:
                # Dictionary of dataframes - export as multiple sheets
                with pd.ExcelWriter(output_path) as writer:
                    for table_name, df in data.items():
                        df.to_excel(writer, sheet_name=table_name, index=False)

            return True

        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return False


class ZipExportAdapter(ExportAdapter):
    """
    Exports data to zipped CSV files
    """

    def export(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
               output_path: str) -> bool:
        """
        Export data to zipped CSV files

        Parameters:
        data: DataFrame or dictionary of DataFrames
        output_path: Path to output zip file

        Returns True if successful
        """
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if isinstance(data, pd.DataFrame):
                    # Single dataframe
                    csv_data = data.to_csv(index=False).encode('utf-8')
                    zipf.writestr('data.csv', csv_data)
                else:
                    # Dictionary of dataframes
                    for table_name, df in data.items():
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        zipf.writestr(f"{table_name}.csv", csv_data)

            return True

        except Exception as e:
            print(f"Error exporting to zip: {e}")
            return False


class ExportFactory:
    """
    Factory for creating export adapters
    """

    @staticmethod
    def create_adapter(format_type: str) -> ExportAdapter:
        """
        Create appropriate export adapter for specified format

        Parameters:
        format_type: Type of format ('csv', 'json', 'parquet', 'sqlite', 'excel', 'zip')

        Returns export adapter
        """
        format_type = format_type.lower()

        if format_type == 'csv':
            return CSVExportAdapter()
        elif format_type == 'json':
            return JSONExportAdapter()
        elif format_type == 'parquet':
            return ParquetExportAdapter()
        elif format_type == 'sqlite':
            return SQLiteExportAdapter()
        elif format_type == 'excel':
            return ExcelExportAdapter()
        elif format_type == 'zip':
            return ZipExportAdapter()
        else:
            raise ValueError(f"Unsupported format type: {format_type}")