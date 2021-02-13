from dataset_utils.build_dataset import BuildDataset
import yaml

if __name__ == "__main__":
    with open("dataset_utils/meta.yaml", 'r') as stream:
        try:
            loader = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    b = BuildDataset(root_directory=loader["root"], mapping=loader["mapping"], urls_csv=loader["urls_csv"],
                     urls_dataset=loader["urls_dataset"], dataset_meta=loader["dataset_meta"],
                     test_specials= loader["test_specials"], label_list=loader["labels"])
