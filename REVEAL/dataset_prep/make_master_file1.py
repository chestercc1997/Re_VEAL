import pandas as pd
import os.path as osp
import os
import argparse

root_folder = osp.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def make_master1(design_names, num_class=5):
    dataset_dict = {}

    for design_name in design_names:
        dataset_dict[design_name] = {
            'num tasks': 1,
            'num classes': num_class,
            'eval metric': 'acc',
            'task type': 'multiclass classification',
            'download_name': design_name,
            'version': 1,
            'url': None,
            'add_inverse_edge': False,
            'has_node_attr': True,
            'has_edge_attr': False,
            'split': 'Random',
            'additional node files': 'None',
            'additional edge files': 'None',
            'is hetero': False,
            'binary': False
        }

    master_dir = osp.join(root_folder, 'dataset_prep', 'master.csv')

    df = pd.DataFrame(dataset_dict)
    df.to_csv(master_dir)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--design_names', type=str, nargs='+', default=[])
    parser.add_argument('--num_class', type=int, default=5)
    args = parser.parse_args()

    make_master1(args.design_names, args.num_class)

if __name__ == "__main__":
    main()