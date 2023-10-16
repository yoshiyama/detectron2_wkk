# register_cusstom_dataset.py
# 以下は、COCO形式のデータセットをDetectron2に登録するためのサンプルPythonファイルの内容です：
# このregister_dataset.pyファイルを実行することで、カスタムデータセットがDetectron2に登録されます。dataset_root_pathを適切なディレクトリに変更することで、あなたの環境に合わせてデータセットの場所を指定することができます。
#
# このスクリプトを実行した後、train_net.pyを実行してトレーニングを開始することができます。

from detectron2.data.datasets import register_coco_instances
import os


def register_custom_datasets(dataset_root):
    """
    Register custom datasets in COCO format.

    Parameters:
    - dataset_root (str): Root directory where the datasets are stored.
    """

    # Train dataset
    train_image_dir = os.path.join(dataset_root, "train/images")
    train_annotation_file = os.path.join(dataset_root, "train/annotations.json")
    register_coco_instances("custom_dataset_train", {}, train_annotation_file, train_image_dir)

    # Validation dataset
    val_image_dir = os.path.join(dataset_root, "val/images")
    val_annotation_file = os.path.join(dataset_root, "val/annotations.json")
    register_coco_instances("custom_dataset_val", {}, val_annotation_file, val_image_dir)


if __name__ == "__main__":
    # Assuming datasets are stored in a directory named "datasets" in the current directory
    dataset_root_path = "./datasets"

    # Register the datasets
    register_custom_datasets(dataset_root_path)
    print("Datasets registered!")
