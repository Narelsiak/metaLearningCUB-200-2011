from .db_controller import DatabaseController
import pandas as pd
import random

class DataLoader:
    def __init__(self):
        self.db = DatabaseController()
    def fetch_table_as_df(self, table_name):
        data = self.db.get_all_data(table_name)
        columns = self.db.get_columns(table_name)
        return pd.DataFrame(data, columns=columns) if data else pd.DataFrame(columns=columns)
    def load_and_merge_data(self):
        df_images = self.fetch_table_as_df("images")
        df_classes = self.fetch_table_as_df("classes")
        df_labels = self.fetch_table_as_df("image_class_labels")
        df_split = self.fetch_table_as_df("train_test_split")

        if df_images.empty or df_classes.empty or df_labels.empty or df_split.empty:
            print("⚠️ Data missing from one of the tables!")
            return None

        df = df_images.merge(df_labels, left_on="id", right_on="image_id")
        df = df.merge(df_classes, left_on="class_id", right_on="id", suffixes=("", "_class"))
        df = df.merge(df_split, on="image_id")

        df.drop(columns=["id_class"], inplace=True, errors="ignore")

        return df
    
    def get_all_class_ids(self):
        df__classes = self.fetch_table_as_df("classes")
        return df__classes["id"].tolist() if not df__classes.empty else []
    
    def load_data_for_classes(self, class_ids):
        df_images = self.fetch_table_as_df("images")
        df_classes = self.fetch_table_as_df("classes")
        df_labels = self.fetch_table_as_df("image_class_labels")
        df_split = self.fetch_table_as_df("train_test_split")

        if df_images.empty or df_classes.empty or df_labels.empty or df_split.empty:
            print("⚠️ Data missing from one of the tables!")
            return None

        df_classes = df_classes[df_classes["id"].isin(class_ids)]
        df_labels = df_labels[df_labels["class_id"].isin(class_ids)]

        df = df_images.merge(df_labels, left_on="id", right_on="image_id")
        df = df.merge(df_classes, left_on="class_id", right_on="id", suffixes=("", "_class"))
        df = df.merge(df_split, on="image_id")
        df.drop(columns=["id_class"], inplace=True, errors="ignore")

        return df