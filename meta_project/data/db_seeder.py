import os 
import pandas as pd
from db_controller import DatabaseController

db = DatabaseController()
db.drop_all_tables()

db.create_table("images", {
    "id": "INTEGER PRIMARY KEY",
    "image_name": "TEXT NOT NULL UNIQUE"
})

db.create_table("classes", {
    "id": "INTEGER PRIMARY KEY",
    "class_name": "TEXT NOT NULL UNIQUE"
})

db.create_table("image_class_labels", {
    "image_id": "INTEGER NOT NULL",
    "class_id": "INTEGER NOT NULL",    
    "PRIMARY KEY (image_id)": "",
    "FOREIGN KEY (image_id)": "REFERENCES images(id) ON DELETE CASCADE",
    "FOREIGN KEY (class_id)": "REFERENCES classes(id) ON DELETE CASCADE"
    })

db.create_table("train_test_split", {
    "image_id": "INTEGER NOT NULL PRIMARY KEY",
    "is_training_image": "INTEGER NOT NULL CHECK(is_training_image IN (0, 1))",
    "FOREIGN KEY (image_id)": "REFERENCES images(id) ON DELETE CASCADE"
    })

data_dir = os.path.join(os.getcwd(), 'data/raw/CUB_200_2011')

def load_data_from_txt(file_name, delimiter=' ', dtype=None):
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path, sep=delimiter, header=None, dtype=dtype)
    return df

def load_data_from_txt_with_id(file_name, delimiter=' ', dtype=None):
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path, sep=delimiter, header=None, dtype=dtype)
    df.insert(0, 'id', range(1, len(df) + 1))
    df.columns = [str(i) for i in range(len(df.columns))]
    return df

def insert_data_from_dataframe(data, table_name):
    for _, row in data.iterrows():
        data = {f"col{i}": row[i] for i in range(len(row))}
        db.insert_data(table_name, data)

images_file = os.path.join(data_dir, 'images.txt')
classes_file = os.path.join(data_dir, 'classes.txt')
image_class_labels_file = os.path.join(data_dir, 'image_class_labels.txt')
train_test_split_file = os.path.join(data_dir, 'train_test_split.txt')

image_data = load_data_from_txt(images_file, delimiter=' ', dtype={0: int, 1: str})
classes_data = load_data_from_txt(classes_file, delimiter=' ', dtype={0: int, 1: str})
image_class_labels_data = load_data_from_txt(image_class_labels_file, delimiter=' ', dtype={0: int, 1: int})
train_test_split_data = load_data_from_txt(train_test_split_file, delimiter=' ', dtype={0: int, 1: int})

insert_data_from_dataframe(image_data, "images")
insert_data_from_dataframe(classes_data, "classes")
insert_data_from_dataframe(image_class_labels_data, "image_class_labels")
insert_data_from_dataframe(train_test_split_data, "train_test_split")
