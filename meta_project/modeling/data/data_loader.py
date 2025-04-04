from db_controller import DatabaseController
import pandas as pd

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

data_loader = DataLoader()
df_combined = data_loader.load_and_merge_data()

print(df_combined.head())